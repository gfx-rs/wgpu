use crate::context::{Context, DynContext};
use crate::{Buffer, Data, Label, C};
use std::sync::Arc;
use std::thread;
use wgt::WasmNotSendSync;

/// Descriptor for the size defining attributes of a triangle geometry, for a bottom level acceleration structure.
pub type BlasTriangleGeometrySizeDescriptor = wgt::BlasTriangleGeometrySizeDescriptor;
static_assertions::assert_impl_all!(BlasTriangleGeometrySizeDescriptor: Send, Sync);

/// Descriptor for the size defining attributes, for a bottom level acceleration structure.
pub type BlasGeometrySizeDescriptors = wgt::BlasGeometrySizeDescriptors;
static_assertions::assert_impl_all!(BlasGeometrySizeDescriptors: Send, Sync);

/// Flags for an acceleration structure.
pub type AccelerationStructureFlags = wgt::AccelerationStructureFlags;
static_assertions::assert_impl_all!(AccelerationStructureFlags: Send, Sync);

/// Flags for a geometry inside a bottom level acceleration structure.
pub type AccelerationStructureGeometryFlags = wgt::AccelerationStructureGeometryFlags;
static_assertions::assert_impl_all!(AccelerationStructureGeometryFlags: Send, Sync);

/// Update mode for acceleration structure builds.
pub type AccelerationStructureUpdateMode = wgt::AccelerationStructureUpdateMode;
static_assertions::assert_impl_all!(AccelerationStructureUpdateMode: Send, Sync);

/// Descriptor to create bottom level acceleration structures.
pub type CreateBlasDescriptor<'a> = wgt::CreateBlasDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CreateBlasDescriptor<'_>: Send, Sync);

/// Safe instance for a [Tlas].
///
/// A TlasInstance may be made invalid, if a TlasInstance is invalid, any attempt to build a [TlasPackage] containing an
/// invalid TlasInstance will generate a validation error
///
/// Each one contains:
/// - A reference to a BLAS, this ***must*** be interacted with using [TlasInstance::new] or [TlasInstance::set_blas], a
/// TlasInstance that references a BLAS keeps that BLAS from being dropped, but if the BLAS is explicitly destroyed (e.g.
/// using [Blas::destroy]) the TlasInstance becomes invalid
/// - A user accessible transformation matrix
/// - A user accessible mask
/// - A user accessible custom index
///
/// [Tlas]: crate::Tlas
/// [TlasPackage]: crate::TlasPackage
#[derive(Debug, Clone)]
pub struct TlasInstance {
    pub(crate) blas: Arc<BlasShared>,
    /// Affine transform matrix 3x4 (rows x columns, row major order).
    pub transform: [f32; 12],
    /// Custom index for the instance used inside the shader.
    ///
    /// This must only use the lower 24 bits, if any bits are outside that range (byte 4 does not equal 0) the TlasInstance becomes
    /// invalid and generates a validation error when built
    pub custom_index: u32,
    /// Mask for the instance used inside the shader to filter instances.
    /// Reports hit only if `(shader_cull_mask & tlas_instance.mask) != 0u`.
    pub mask: u8,
}

impl TlasInstance {
    /// Construct TlasInstance.
    /// - blas: Reference to the bottom level acceleration structure
    /// - transform: Transform buffer offset in bytes (optional, required if transform buffer is present)
    /// - custom_index: Custom index for the instance used inside the shader (max 24 bits)
    /// - mask: Mask for the instance used inside the shader to filter instances
    ///
    /// Note: while one of these contains a reference to a BLAS that BLAS will not be dropped,
    /// but it can still be destroyed. Destroying a BLAS that is referenced by one or more
    /// TlasInstance(s) will immediately make them invalid. If one or more of those invalid
    /// TlasInstances is inside a TlasPackage that is attempted to be built, the build will
    /// generate a validation error.
    pub fn new(blas: &Blas, transform: [f32; 12], custom_index: u32, mask: u8) -> Self {
        Self {
            blas: blas.shared.clone(),
            transform,
            custom_index,
            mask,
        }
    }

    /// Set the bottom level acceleration structure.
    ///
    /// See the note on [TlasInstance] about the
    /// guarantees of keeping a BLAS alive.
    pub fn set_blas(&mut self, blas: &Blas) {
        self.blas = blas.shared.clone();
    }
}

pub(crate) struct DynContextTlasInstance<'a> {
    pub(crate) blas: &'a Data,
    pub(crate) transform: &'a [f32; 12],
    pub(crate) custom_index: u32,
    pub(crate) mask: u8,
}

/// Context version of [TlasInstance].
#[allow(dead_code)]
pub struct ContextTlasInstance<'a, T: Context> {
    pub(crate) blas_data: &'a T::BlasData,
    pub(crate) transform: &'a [f32; 12],
    pub(crate) custom_index: u32,
    pub(crate) mask: u8,
}

#[derive(Debug)]
/// Definition for a triangle geometry for a Bottom Level Acceleration Structure (BLAS).
///
/// The size must match the rest of the structures fields, otherwise the build will fail.
/// (e.g. if index count is present in the size, the index buffer must be present as well.)
pub struct BlasTriangleGeometry<'a> {
    /// Sub descriptor for the size defining attributes of a triangle geometry.
    pub size: &'a BlasTriangleGeometrySizeDescriptor,
    /// Vertex buffer.
    pub vertex_buffer: &'a Buffer,
    /// Offset into the vertex buffer as a factor of the vertex stride.
    pub first_vertex: u32,
    /// Vertex stride.
    pub vertex_stride: wgt::BufferAddress,
    /// Index buffer (optional).
    pub index_buffer: Option<&'a Buffer>,
    /// Index buffer offset in bytes (optional, required if index buffer is present).
    pub index_buffer_offset: Option<wgt::BufferAddress>,
    /// Transform buffer containing 3x4 (rows x columns, row major) affine transform matrices `[f32; 12]` (optional).
    pub transform_buffer: Option<&'a Buffer>,
    /// Transform buffer offset in bytes (optional, required if transform buffer is present).
    pub transform_buffer_offset: Option<wgt::BufferAddress>,
}
static_assertions::assert_impl_all!(BlasTriangleGeometry<'_>: WasmNotSendSync);

/// Contains the sets of geometry that go into a [Blas].
pub enum BlasGeometries<'a> {
    /// Triangle geometry variant.
    TriangleGeometries(Vec<BlasTriangleGeometry<'a>>),
}
static_assertions::assert_impl_all!(BlasGeometries<'_>: WasmNotSendSync);

/// Builds the given sets of geometry into the given [Blas].
pub struct BlasBuildEntry<'a> {
    /// Reference to the acceleration structure.
    pub blas: &'a Blas,
    /// Geometries.
    pub geometry: BlasGeometries<'a>,
}
static_assertions::assert_impl_all!(BlasBuildEntry<'_>: WasmNotSendSync);

#[derive(Debug)]
pub(crate) struct BlasShared {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
static_assertions::assert_impl_all!(BlasShared: WasmNotSendSync);

#[derive(Debug)]
/// Bottom Level Acceleration Structure (BLAS).
///
/// A BLAS is a device-specific raytracing acceleration structure that contains geometry data.
///
/// These BLASes are combined with transform in a [TlasInstance] to create a [Tlas].
///
/// [Tlas]: crate::Tlas
pub struct Blas {
    pub(crate) handle: Option<u64>,
    pub(crate) shared: Arc<BlasShared>,
}
static_assertions::assert_impl_all!(Blas: WasmNotSendSync);

impl Blas {
    /// Raw handle to the acceleration structure, used inside raw instance buffers.
    pub fn handle(&self) -> Option<u64> {
        self.handle
    }
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::blas_destroy(&*self.shared.context, self.shared.data.as_ref());
    }
}

impl Drop for BlasShared {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.blas_drop(self.data.as_ref());
        }
    }
}

pub(crate) struct DynContextBlasTriangleGeometry<'a> {
    pub(crate) size: &'a BlasTriangleGeometrySizeDescriptor,
    pub(crate) vertex_buffer: &'a Data,
    pub(crate) index_buffer: Option<&'a Data>,
    pub(crate) transform_buffer: Option<&'a Data>,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_stride: wgt::BufferAddress,
    pub(crate) index_buffer_offset: Option<wgt::BufferAddress>,
    pub(crate) transform_buffer_offset: Option<wgt::BufferAddress>,
}

pub(crate) enum DynContextBlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = DynContextBlasTriangleGeometry<'a>> + 'a>),
}

pub(crate) struct DynContextBlasBuildEntry<'a> {
    pub(crate) blas_data: &'a Data,
    pub(crate) geometries: DynContextBlasGeometries<'a>,
}

/// Context version of [BlasTriangleGeometry].
#[allow(dead_code)]
pub struct ContextBlasTriangleGeometry<'a, T: Context> {
    pub(crate) size: &'a BlasTriangleGeometrySizeDescriptor,
    pub(crate) vertex_buffer: &'a T::BufferData,
    pub(crate) index_buffer: Option<&'a T::BufferData>,
    pub(crate) transform_buffer: Option<&'a T::BufferData>,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_stride: wgt::BufferAddress,
    pub(crate) index_buffer_offset: Option<wgt::BufferAddress>,
    pub(crate) transform_buffer_offset: Option<wgt::BufferAddress>,
}

/// Context version of [BlasGeometries].
pub enum ContextBlasGeometries<'a, T: Context> {
    /// Triangle geometries.
    TriangleGeometries(Box<dyn Iterator<Item = ContextBlasTriangleGeometry<'a, T>> + 'a>),
}

/// Context version see [BlasBuildEntry].
#[allow(dead_code)]
pub struct ContextBlasBuildEntry<'a, T: Context> {
    pub(crate) blas_data: &'a T::BlasData,
    pub(crate) geometries: ContextBlasGeometries<'a, T>,
}
