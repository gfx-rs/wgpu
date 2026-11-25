use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{id, Label};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum BuildPreference {
    #[default]
    FastTrace,
    FastBuild,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleGeometrySize {
    pub vertex_count: u32,
    pub index: Option<TriangleIndexSize>,
    pub opaque: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TriangleIndexSize {
    pub format: wgt::IndexFormat,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlasDescriptor<'a> {
    pub label: Label<'a>,
    pub geometries: Vec<TriangleGeometrySize>,
    pub build_preference: BuildPreference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TlasDescriptor<'a> {
    pub label: Label<'a>,
    pub max_instances: u32,
    pub build_preference: BuildPreference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleGeometry {
    pub size: TriangleGeometrySize,
    pub vertex_buffer: id::BufferId,
    pub first_vertex: u32,
    pub vertex_stride: wgt::BufferAddress,
    pub index_buffer: Option<id::BufferId>,
    pub first_index: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlasBuild {
    pub blas: id::BlasId,
    pub geometries: Vec<TriangleGeometry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TlasInstance {
    pub blas: id::BlasId,
    pub transform: [f32; 12],
    pub custom_index: u32,
    pub mask: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TlasBuild {
    pub tlas: id::TlasId,
    pub instances: Vec<TlasInstance>,
}
