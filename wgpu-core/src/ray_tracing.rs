use crate::{
    command::CommandEncoderError,
    id::{BlasId, BufferId, TlasId},
};

use thiserror::Error;
use wgt::BufferAddress;

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),

    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),

    #[error("Buffer {0:?} is missing `BLAS_INPUT` usage flag")]
    MissingBlasInputUsageFlag(BufferId),

    #[error(
        "Buffer {0:?} size is insufficient for provided size information (size: {1}, required: {2}"
    )]
    InsufficientBufferSize(BufferId, u64, u64),

    #[error("Buffer {0:?} associated vertex buffer sizes invalid (no vertices)")]
    EmptyVertexBuffer(BufferId),

    #[error("Buffer {0:?} associated index buffer sizes invalid (less then three indices)")]
    EmptyIndexBuffer(BufferId),

    #[error("Buffer {0:?} associated index count not divisible by 3 (count: {1}")]
    InvalidIndexCount(BufferId, u32),

    #[error("Buffer {0:?} associated data contains None")]
    MissingAssociatedData(BufferId),

    #[error(
        "Blas {0:?} build sizes to may be greater than the descriptor at build time specified"
    )]
    IncompatibleBlasBuildSizes(BlasId),

    #[error("Blas {0:?} is invalid or destroyed")]
    InvalidBlas(BlasId),

    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(TlasId),

    #[error("Buffer {0:?} is missing `TLAS_INPUT` usage flag")]
    MissingTlasInputUsageFlag(BufferId),
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

pub struct TlasBuildEntry {
    pub tlas_id: TlasId,
    pub instance_buffer_id: BufferId,
    pub instance_count: u32,
}
