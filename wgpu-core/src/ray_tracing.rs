// Ray tracing
// Major missing optimizations (no api surface changes needed):
// - use custom tracker to track build state
// - no forced rebuilt (build mode deduction)
// - lazy instance buffer allocation
// - maybe share scratch and instance staging buffer allocation
// - partial instance buffer uploads (api surface already designed with this in mind)
// - ([non performance] extract function in build (rust function extraction with guards is a pain))

use crate::{
    command::CommandEncoderError,
    device::DeviceError,
    id::{BlasId, BufferId, TlasId},
    resource::CreateBufferError,
};
use std::num::NonZeroU64;
use std::sync::Arc;

use crate::resource::{Blas, ResourceErrorIdent, Tlas};
use thiserror::Error;
use wgt::{AccelerationStructureGeometryFlags, BufferAddress, IndexFormat, VertexFormat};

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
    #[error("Provided format was not within allowed formats. Provided format: {0:?}. Allowed formats: {1:?}")]
    InvalidVertexFormat(VertexFormat, Vec<VertexFormat>),
    #[error("Features::RAY_TRACING_ACCELERATION_STRUCTURE is not enabled")]
    MissingFeature,
}

#[derive(Clone, Debug, Error)]
pub enum CreateTlasError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateBufferError(#[from] CreateBufferError),
    #[error("Features::RAY_TRACING_ACCELERATION_STRUCTURE is not enabled")]
    MissingFeature,
}

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error("BufferId is invalid or destroyed")]
    InvalidBufferId,

    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(ResourceErrorIdent),

    #[error("Buffer {0:?} is missing `BLAS_INPUT` usage flag")]
    MissingBlasInputUsageFlag(ResourceErrorIdent),

    #[error(
        "Buffer {0:?} size is insufficient for provided size information (size: {1}, required: {2}"
    )]
    InsufficientBufferSize(ResourceErrorIdent, u64, u64),

    #[error("Buffer {0:?} associated offset doesn't align with the index type")]
    UnalignedIndexBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated offset is unaligned")]
    UnalignedTransformBufferOffset(ResourceErrorIdent),

    #[error("Buffer {0:?} associated index count not divisible by 3 (count: {1}")]
    InvalidIndexCount(ResourceErrorIdent, u32),

    #[error("Buffer {0:?} associated data contains None")]
    MissingAssociatedData(ResourceErrorIdent),

    #[error(
        "Blas {0:?} build sizes to may be greater than the descriptor at build time specified"
    )]
    IncompatibleBlasBuildSizes(ResourceErrorIdent),

    #[error("Blas {0:?} flags are different, creation flags: {1:?}, provided: {2:?}")]
    IncompatibleBlasFlags(
        ResourceErrorIdent,
        AccelerationStructureGeometryFlags,
        AccelerationStructureGeometryFlags,
    ),

    #[error("Blas {0:?} build vertex count is greater than creation count (needs to be less than or equal to), creation: {1:?}, build: {2:?}")]
    IncompatibleBlasVertexCount(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} vertex formats are different, creation format: {1:?}, provided: {2:?}")]
    DifferentBlasVertexFormats(ResourceErrorIdent, VertexFormat, VertexFormat),

    #[error("Blas {0:?} index count was provided at creation or building, but not the other")]
    BlasIndexCountProvidedMismatch(ResourceErrorIdent),

    #[error("Blas {0:?} build index count is greater than creation count (needs to be less than or equal to), creation: {1:?}, build: {2:?}")]
    IncompatibleBlasIndexCount(ResourceErrorIdent, u32, u32),

    #[error("Blas {0:?} index formats are different, creation format: {1:?}, provided: {2:?}")]
    DifferentBlasIndexFormats(ResourceErrorIdent, Option<IndexFormat>, Option<IndexFormat>),

    #[error("Blas {0:?} build sizes require index buffer but none was provided")]
    MissingIndexBuffer(ResourceErrorIdent),

    #[error("BlasId is invalid")]
    InvalidBlasId,

    #[error("Blas {0:?} is destroyed")]
    InvalidBlas(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} an associated instances contains an invalid custom index (more than 24bits)"
    )]
    TlasInvalidCustomIndex(ResourceErrorIdent),

    #[error(
        "Tlas {0:?} has {1} active instances but only {2} are allowed as specified by the descriptor at creation"
    )]
    TlasInstanceCountExceeded(ResourceErrorIdent, u32, u32),

    #[error("BlasId is invalid or destroyed (for instance)")]
    InvalidBlasIdForInstance,

    #[error("TlasId is invalid or destroyed")]
    InvalidTlasId,

    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(ResourceErrorIdent),

    #[error("Features::RAY_TRACING_ACCELERATION_STRUCTURE is not enabled")]
    MissingFeature,

    #[error("Buffer {0:?} is missing `TLAS_INPUT` usage flag")]
    MissingTlasInputUsageFlag(ResourceErrorIdent),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateBlasActionsError {
    #[error("Blas {0:?} is used before it is built")]
    UsedUnbuilt(ResourceErrorIdent),
}

#[derive(Clone, Debug, Error)]
pub enum ValidateTlasActionsError {
    #[error("Tlas {0:?} is used before it is built")]
    UsedUnbuilt(ResourceErrorIdent),

    #[error("Blas {0:?} is used before it is built (in Tlas {1:?})")]
    UsedUnbuiltBlas(ResourceErrorIdent, ResourceErrorIdent),

    #[error("BlasId is destroyed (in Tlas {0:?})")]
    InvalidBlas(ResourceErrorIdent),

    #[error("Blas {0:?} is newer than the containing Tlas {1:?}")]
    BlasNewerThenTlas(ResourceErrorIdent, ResourceErrorIdent),
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
        dependencies: Vec<Arc<Blas>>,
    },
    Use,
}

#[derive(Debug, Clone)]
pub(crate) struct BlasAction {
    pub blas: Arc<Blas>,
    pub kind: BlasActionKind,
}

#[derive(Debug, Clone)]
pub(crate) struct TlasAction {
    pub tlas: Arc<Tlas>,
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
