/// Ray tracing
/// Major missing optimizations (no api surface changes needed):
/// - use custom tracker to track build state
/// - no forced rebuilt (build mode deduction)
/// - lazy instance buffer allocation
/// - maybe share scratch and instance staging buffer allocation
/// - partial instance buffer uploads (api surface already designed with this in mind)
/// - ([non performance] extract function in build (rust function extraction with guards is a pain))
use std::{num::NonZeroU64, slice};

use crate::{
    command::CommandEncoderError,
    device::DeviceError,
    hal_api::HalApi,
    id::{BlasId, BufferId, TlasId},
    resource::CreateBufferError,
};

use thiserror::Error;
use wgt::BufferAddress;

#[derive(Clone, Debug, Error)]
pub enum CreateBlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateBufferError(#[from] CreateBufferError),
    #[error(
        "Only one of 'index_count' and 'index_format' was provided (either provide both or none)"
    )]
    MissingIndexData,
}

#[derive(Clone, Debug, Error)]
pub enum CreateTlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateBufferError(#[from] CreateBufferError),
    #[error("Unimplemented Tlas error: this error is not yet implemented")]
    Unimplemented,
}

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),

    #[error("Buffer {0:?} is missing `BLAS_INPUT` usage flag")]
    MissingBlasInputUsageFlag(BufferId),

    #[error(
        "Buffer {0:?} size is insufficient for provided size information (size: {1}, required: {2}"
    )]
    InsufficientBufferSize(BufferId, u64, u64),

    #[error("Buffer {0:?} associated offset doesn't align with the index type")]
    UnalignedIndexBufferOffset(BufferId),

    #[error("Buffer {0:?} associated offset is unaligned")]
    UnalignedTransformBufferOffset(BufferId),

    #[error("Buffer {0:?} associated index count not divisible by 3 (count: {1}")]
    InvalidIndexCount(BufferId, u32),

    #[error("Buffer {0:?} associated data contains None")]
    MissingAssociatedData(BufferId),

    #[error(
        "Blas {0:?} build sizes to may be greater than the descriptor at build time specified"
    )]
    IncompatibleBlasBuildSizes(BlasId),

    #[error("Blas {0:?} build sizes require index buffer but none was provided")]
    MissingIndexBuffer(BlasId),

    #[error("Blas {0:?} is invalid or destroyed")]
    InvalidBlas(BlasId),

    #[error(
        "Tlas {0:?} an associated instances contains an invalid custom index (more than 24bits)"
    )]
    TlasInvalidCustomIndex(TlasId),

    #[error(
        "Tlas {0:?} has {1} active instances but only {2} are allowed as specified by the descriptor at creation"
    )]
    TlasInstanceCountExceeded(TlasId, u32, u32),

    #[error("Blas {0:?} is invalid or destroyed (for instance)")]
    InvalidBlasForInstance(BlasId),

    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(TlasId),

    #[error("Buffer {0:?} is missing `TLAS_INPUT` usage flag")]
    MissingTlasInputUsageFlag(BufferId),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateBlasActionsError {
    #[error("Blas {0:?} is invalid or destroyed")]
    InvalidBlas(BlasId),

    #[error("Blas {0:?} is used before it is build")]
    UsedUnbuilt(BlasId),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateTlasActionsError {
    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(TlasId),

    #[error("Tlas {0:?} is used before it is build")]
    UsedUnbuilt(TlasId),

    #[error("Blas {0:?} is used before it is build (in Tlas {1:?})")]
    UsedUnbuiltBlas(BlasId, TlasId),

    #[error("Blas {0:?} is invalid or destroyed (in Tlas {1:?})")]
    InvalidBlas(BlasId, TlasId),

    #[error("Blas {0:?} is newer than the containing Tlas {1:?}")]
    BlasNewerThenTlas(BlasId, TlasId),
}

#[derive(Debug)]
pub struct BlasTriangleGeometry<'a> {
    pub size: &'a wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer_offset: Option<BufferAddress>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

pub enum BlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = BlasTriangleGeometry<'a>> + 'a>),
}

pub struct BlasBuildEntry<'a> {
    pub blas_id: BlasId,
    pub geometries: BlasGeometries<'a>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TlasBuildEntry {
    pub tlas_id: TlasId,
    pub instance_buffer_id: BufferId,
    pub instance_count: u32,
}

#[derive(Debug)]
pub struct TlasInstance<'a> {
    pub blas_id: BlasId,
    pub transform: &'a [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
}

pub struct TlasPackage<'a> {
    pub tlas_id: TlasId,
    pub instances: Box<dyn Iterator<Item = Option<TlasInstance<'a>>> + 'a>,
    pub lowest_unmodified: u32,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum BlasActionKind {
    Build(NonZeroU64),
    Use,
}

#[derive(Debug, Clone)]
pub(crate) enum TlasActionKind {
    Build {
        build_index: NonZeroU64,
        dependencies: Vec<BlasId>,
    },
    Use,
}

#[derive(Debug, Clone)]
pub(crate) struct BlasAction {
    pub id: BlasId,
    pub kind: BlasActionKind,
}

#[derive(Debug, Clone)]
pub(crate) struct TlasAction {
    pub id: TlasId,
    pub kind: TlasActionKind,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceBlasTriangleGeometry {
    pub size: wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer_offset: Option<BufferAddress>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TraceBlasGeometries {
    TriangleGeometries(Vec<TraceBlasTriangleGeometry>),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceBlasBuildEntry {
    pub blas_id: BlasId,
    pub geometries: TraceBlasGeometries,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceTlasInstance {
    pub blas_id: BlasId,
    pub transform: [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceTlasPackage {
    pub tlas_id: TlasId,
    pub instances: Vec<Option<TraceTlasInstance>>,
    pub lowest_unmodified: u32,
}

pub(crate) fn get_raw_tlas_instance_size<A: HalApi>() -> usize {
    match A::VARIANT {
        wgt::Backend::Empty => 0,
        wgt::Backend::Vulkan => 64,
        _ => unimplemented!(),
    }
}

#[derive(Clone)]
#[repr(C)]
struct RawTlasInstance {
    transform: [f32; 12],
    custom_index_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

pub(crate) fn tlas_instance_into_bytes<A: HalApi>(
    instance: &TlasInstance,
    blas_address: u64,
) -> Vec<u8> {
    match A::VARIANT {
        wgt::Backend::Empty => vec![],
        wgt::Backend::Vulkan => {
            const MAX_U24: u32 = (1u32 << 24u32) - 1u32;
            let temp = RawTlasInstance {
                transform: *instance.transform,
                custom_index_and_mask: (instance.custom_index & MAX_U24)
                    | (u32::from(instance.mask) << 24),
                shader_binding_table_record_offset_and_flags: 0,
                acceleration_structure_reference: blas_address,
            };
            let temp: *const _ = &temp;
            unsafe {
                slice::from_raw_parts::<u8>(
                    temp as *const u8,
                    std::mem::size_of::<RawTlasInstance>(),
                )
                .to_vec()
            }
        }
        _ => unimplemented!(),
    }
}
