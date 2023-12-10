use std::{fmt::Debug, ops::Range, sync::Arc, thread};
use wgt::WasmNotSendSync;

use crate::{
    context::{Context, DynContext, ObjectId},
    BindingResource, Buffer, CommandEncoder, Data, Device, Label, C,
};

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

/// Descriptor to create top level acceleration structures.
pub type CreateTlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CreateTlasDescriptor<'_>: Send, Sync);

#[derive(Debug)]
/// Definition for a triangle geometry.
/// The size must match the rest of the structures fields, otherwise the build will fail.
/// (e.g. if a index count is present in the size, the index buffer must be present as well.)
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
    /// Transform buffer containing 3x4 (rows x columns, row mayor) affine transform matrices `[f32; 12]` (optional).
    pub transform_buffer: Option<&'a Buffer>,
    /// Transform buffer offset in bytes (optional, required if transform buffer is present).
    pub transform_buffer_offset: Option<wgt::BufferAddress>,
}
static_assertions::assert_impl_all!(BlasTriangleGeometry<'_>: WasmNotSendSync);

/// Geometries for a bottom level acceleration structure.
pub enum BlasGeometries<'a> {
    /// Triangle geometry variant.
    TriangleGeometries(Vec<BlasTriangleGeometry<'a>>),
}
static_assertions::assert_impl_all!(BlasGeometries<'_>: WasmNotSendSync);

/// Entry for a bottom level acceleration structure build.
pub struct BlasBuildEntry<'a> {
    /// Reference to the acceleration structure.
    pub blas: &'a Blas,
    /// Geometries.
    pub geometry: BlasGeometries<'a>,
}
static_assertions::assert_impl_all!(BlasBuildEntry<'_>: WasmNotSendSync);

#[derive(Debug)]
/// Bottom level acceleration structure.
/// Used to represent a collection of geometries for ray tracing inside a top level acceleration structure.
pub struct Blas {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
    pub(crate) handle: Option<u64>,
}
static_assertions::assert_impl_all!(Blas: WasmNotSendSync);

impl Blas {
    /// Raw handle to the acceleration structure, used inside raw instance buffers.
    pub fn handle(&self) -> Option<u64> {
        self.handle
    }
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::blas_destroy(&*self.context, &self.id, self.data.as_ref());
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.blas_drop(&self.id, self.data.as_ref());
        }
    }
}

#[derive(Debug)]
/// Top level acceleration structure.
/// Used to represent a collection of bottom level acceleration structure instances for ray tracing.
pub struct Tlas {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
static_assertions::assert_impl_all!(Tlas: WasmNotSendSync);

impl Tlas {
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::tlas_destroy(&*self.context, &self.id, self.data.as_ref());
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.tlas_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Entry for a top level acceleration structure build.
/// Used with raw instance buffers for a unvalidated builds.
pub struct TlasBuildEntry<'a> {
    /// Reference to the acceleration structure.
    pub tlas: &'a Tlas,
    /// Reference to the raw instance buffer.
    pub instance_buffer: &'a Buffer,
    /// Number of instances in the instance buffer.
    pub instance_count: u32,
}
static_assertions::assert_impl_all!(TlasBuildEntry<'_>: WasmNotSendSync);

/// Safe instance for a top level acceleration structure.
#[derive(Debug, Clone)]
pub struct TlasInstance {
    pub(crate) blas: ObjectId,
    /// Affine transform matrix 3x4 (rows x columns, row mayor order).
    pub transform: [f32; 12],
    /// Custom index for the instance used inside the shader (max 24 bits).
    pub custom_index: u32,
    /// Mask for the instance used inside the shader to filter instances.
    pub mask: u8,
}

impl TlasInstance {
    /// Construct TlasInstance.
    /// - blas: Reference to the bottom level acceleration structure
    /// - transform: Transform buffer offset in bytes (optional, required if transform buffer is present)
    /// - custom_index: Custom index for the instance used inside the shader (max 24 bits)
    /// - mask: Mask for the instance used inside the shader to filter instances
    pub fn new(blas: &Blas, transform: [f32; 12], custom_index: u32, mask: u8) -> Self {
        Self {
            blas: blas.id,
            transform,
            custom_index,
            mask,
        }
    }

    /// Set the bottom level acceleration structure.
    pub fn set_blas(&mut self, blas: &Blas) {
        self.blas = blas.id;
    }
}

pub(crate) struct DynContextTlasInstance<'a> {
    pub(crate) blas: ObjectId,
    pub(crate) transform: &'a [f32; 12],
    pub(crate) custom_index: u32,
    pub(crate) mask: u8,
}

/// [Context version] see `TlasInstance`.
pub struct ContextTlasInstance<'a, T: Context> {
    pub(crate) blas_id: T::BlasId,
    pub(crate) transform: &'a [f32; 12],
    pub(crate) custom_index: u32,
    pub(crate) mask: u8,
}

/// The safe version of TlasEntry, containing TlasInstances instead of a raw buffer.
pub struct TlasPackage {
    pub(crate) tlas: Tlas,
    pub(crate) instances: Vec<Option<TlasInstance>>,
    pub(crate) lowest_unmodified: u32,
}
static_assertions::assert_impl_all!(TlasPackage: WasmNotSendSync);

impl TlasPackage {
    /// Construct TlasPackage consuming the Tlas (prevents modification of the Tlas without using this package).
    /// (max_instances needs to fit into tlas)
    pub fn new(tlas: Tlas, max_instances: u32) -> Self {
        Self::new_with_instances(tlas, vec![None; max_instances as usize])
    }

    /// Construct TlasPackage consuming the Tlas (prevents modification of the Tlas without using this package).
    /// This contructor moves the instances into the package (the number of instances needs to fit into tlas).
    pub fn new_with_instances(tlas: Tlas, instances: Vec<Option<TlasInstance>>) -> Self {
        Self {
            tlas,
            lowest_unmodified: instances.len() as u32,
            instances,
        }
    }

    /// Get a reference to all instances.
    pub fn get(&self) -> &[Option<TlasInstance>] {
        &self.instances
    }

    /// Get a mutable slice to a range of instances.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    /// For better performance it is recommended to reduce access to low elements.
    pub fn get_mut_slice(&mut self, range: Range<usize>) -> Option<&mut [Option<TlasInstance>]> {
        if range.end > self.instances.len() {
            return None;
        }
        if range.end as u32 > self.lowest_unmodified {
            self.lowest_unmodified = range.end as u32;
        }
        Some(&mut self.instances[range])
    }

    /// Get a single mutable reference to an instance.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    /// For better performance it is recommended to reduce access to low elements.
    pub fn get_mut_single(&mut self, index: usize) -> Option<&mut Option<TlasInstance>> {
        if index >= self.instances.len() {
            return None;
        }
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        Some(&mut self.instances[index])
    }

    /// Get the binding resource for the underling acceleration structure, to be used in a
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(&self.tlas)
    }

    /// Get a reference to the underling top level acceleration structure.
    pub fn tlas(&self) -> &Tlas {
        &self.tlas
    }
}

pub(crate) struct DynContextBlasTriangleGeometry<'a> {
    pub(crate) size: &'a BlasTriangleGeometrySizeDescriptor,
    pub(crate) vertex_buffer: ObjectId,
    pub(crate) index_buffer: Option<ObjectId>,
    pub(crate) transform_buffer: Option<ObjectId>,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_stride: wgt::BufferAddress,
    pub(crate) index_buffer_offset: Option<wgt::BufferAddress>,
    pub(crate) transform_buffer_offset: Option<wgt::BufferAddress>,
}

pub(crate) enum DynContextBlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = DynContextBlasTriangleGeometry<'a>> + 'a>),
}

pub(crate) struct DynContextBlasBuildEntry<'a> {
    pub(crate) blas_id: ObjectId,
    pub(crate) geometries: DynContextBlasGeometries<'a>,
}

pub(crate) struct DynContextTlasBuildEntry {
    pub(crate) tlas_id: ObjectId,
    pub(crate) instance_buffer_id: ObjectId,
    pub(crate) instance_count: u32,
}

pub(crate) struct DynContextTlasPackage<'a> {
    pub(crate) tlas_id: ObjectId,
    pub(crate) instances: Box<dyn Iterator<Item = Option<DynContextTlasInstance<'a>>> + 'a>,
    pub(crate) lowest_unmodified: u32,
}

/// [Context version] see `BlasTriangleGeometry`.
pub struct ContextBlasTriangleGeometry<'a, T: Context> {
    pub(crate) size: &'a BlasTriangleGeometrySizeDescriptor,
    pub(crate) vertex_buffer: T::BufferId,
    pub(crate) index_buffer: Option<T::BufferId>,
    pub(crate) transform_buffer: Option<T::BufferId>,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_stride: wgt::BufferAddress,
    pub(crate) index_buffer_offset: Option<wgt::BufferAddress>,
    pub(crate) transform_buffer_offset: Option<wgt::BufferAddress>,
}

/// [Context version] see `BlasGeometries`.
pub enum ContextBlasGeometries<'a, T: Context> {
    /// Triangle geometries.
    TriangleGeometries(Box<dyn Iterator<Item = ContextBlasTriangleGeometry<'a, T>> + 'a>),
}

/// [Context version] see `BlasBuildEntry`.
pub struct ContextBlasBuildEntry<'a, T: Context> {
    pub(crate) blas_id: T::BlasId,
    pub(crate) geometries: ContextBlasGeometries<'a, T>,
}

/// [Context version] see `TlasBuildEntry`.
pub struct ContextTlasBuildEntry<T: Context> {
    pub(crate) tlas_id: T::TlasId,
    pub(crate) instance_buffer_id: T::BufferId,
    pub(crate) instance_count: u32,
}

/// [Context version] see `TlasPackage`.
pub struct ContextTlasPackage<'a, T: Context> {
    pub(crate) tlas_id: T::TlasId,
    pub(crate) instances: Box<dyn Iterator<Item = Option<ContextTlasInstance<'a, T>>> + 'a>,
    pub(crate) lowest_unmodified: u32,
}

/// Utility module to add traits for the device and command encoder.    
pub mod traits {
    pub use super::{CommandEncoderRayTracing as _, DeviceRayTracing as _};
}

/// Trait to add ray tracing functions to a [`Device`].
pub trait DeviceRayTracing {
    /// Create a bottom level acceleration structure, used inside a top level acceleration structure for ray tracing.
    /// - desc: The descriptor of the acceleration structure.
    /// - sizes: Size descriptor limiting what can be built into the acceleration structure.
    fn create_blas(&self, desc: &CreateBlasDescriptor<'_>, sizes: BlasGeometrySizeDescriptors) -> Blas;

    /// Create a top level acceleration structure, used for ray tracing.
    /// - desc: The descriptor of the acceleration structure.
    fn create_tlas(&self, desc: &CreateTlasDescriptor<'_>) -> Tlas;
}

impl DeviceRayTracing for Device {
    fn create_blas(&self, desc: &CreateBlasDescriptor<'_>, sizes: BlasGeometrySizeDescriptors) -> Blas {
        let (id, handle, data) = DynContext::device_create_blas(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
            sizes,
        );

        Blas {
            context: Arc::clone(&self.context),
            id,
            data,
            handle,
        }
    }

    fn create_tlas(&self, desc: &CreateTlasDescriptor<'_>) -> Tlas {
        let (id, data) =
            DynContext::device_create_tlas(&*self.context, &self.id, self.data.as_ref(), desc);

        Tlas {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }
}

/// Trait to add ray tracing functions to a [`CommandEncoder`].
pub trait CommandEncoderRayTracing {
    /// Build bottom and top level acceleration structures.
    /// - blas: Iterator of bottom level acceleration structure entries to build
    ///     For each entry, the provided size descriptor must be strictly smaller or equal to the descriptor given at bottom level acceleration structure creation:
    ///     - Less or equal number of geometries
    ///     - Same kind of geometry (with index buffer or without) (same vertex/index format)
    ///     - Same flags
    ///     - Less or equal number of vertices
    ///     - Less or equal number of indices (if applicable)
    /// - tlas: iterator of top level acceleration structure packages to build
    ///
    /// A bottom level acceleration structure may be build and used as a reference in a top level acceleration structure in the same invocation of this function.
    ///
    /// # Bind group usage
    ///
    /// When a top level acceleration structure is used in a bind group, some validation takes place:
    ///    - The top level acceleration structure is valid and has been built.
    ///    - All the bottom level acceleration structures referenced by the top level acceleration structure are valid and have been built prior,
    ///      or at same time as the containing top level acceleration structure.
    fn build_acceleration_structures<'a>(
        &mut self,
        blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>,
        tlas: impl IntoIterator<Item = &'a TlasPackage>,
    );

    /// Build bottom and top level acceleration structures.
    /// See [`build_acceleration_structures`] for the safe version and more details.
    ///
    /// # Safety
    ///
    ///    - The contents of the raw instance buffer must be valid for the underling api.
    ///    - All bottom level acceleration structures, referenced in the raw instance buffer must be valid and built,
    ///       when the corresponding top level acceleration structure is built. (builds may happen in the same invocation of this function).
    ///    - At the time when the top level acceleration structure is used in a bind group, all associated bottom level acceleration structures must be valid,
    ///      and built (no later than the time when the top level acceleration structure was built).
    unsafe fn build_acceleration_structures_unsafe_tlas<'a>(
        &mut self,
        blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>,
        tlas: impl IntoIterator<Item = &'a TlasBuildEntry<'a>>,
    );
}

impl CommandEncoderRayTracing for CommandEncoder {
    fn build_acceleration_structures<'a>(
        &mut self,
        blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>,
        tlas: impl IntoIterator<Item = &'a TlasPackage>,
    ) {
        let id = self.id.as_ref().unwrap();

        let mut blas = blas.into_iter().map(|e: &BlasBuildEntry<'_>| {
            let geometries = match &e.geometry {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg: &BlasTriangleGeometry<'_>| {
                        DynContextBlasTriangleGeometry {
                            size: tg.size,
                            vertex_buffer: tg.vertex_buffer.id,

                            index_buffer: tg.index_buffer.map(|index_buffer| index_buffer.id),

                            transform_buffer: tg
                                .transform_buffer
                                .map(|transform_buffer| transform_buffer.id),

                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                            transform_buffer_offset: tg.transform_buffer_offset,
                        }
                    });
                    DynContextBlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            DynContextBlasBuildEntry {
                blas_id: e.blas.id,
                geometries,
            }
        });

        let mut tlas = tlas.into_iter().map(|e: &TlasPackage| {
            let instances = e.instances.iter().map(|instance: &Option<TlasInstance>| {
                instance.as_ref().map(|instance| DynContextTlasInstance {
                    blas: instance.blas,
                    transform: &instance.transform,
                    custom_index: instance.custom_index,
                    mask: instance.mask,
                })
            });
            DynContextTlasPackage {
                tlas_id: e.tlas.id,
                instances: Box::new(instances),
                lowest_unmodified: e.lowest_unmodified,
            }
        });

        DynContext::command_encoder_build_acceleration_structures(
            &*self.context,
            id,
            self.data.as_ref(),
            &mut blas,
            &mut tlas,
        );
    }

    unsafe fn build_acceleration_structures_unsafe_tlas<'a>(
        &mut self,
        blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>,
        tlas: impl IntoIterator<Item = &'a TlasBuildEntry<'a>>,
    ) {
        let id = self.id.as_ref().unwrap();

        let mut blas = blas.into_iter().map(|e: &BlasBuildEntry<'_>| {
            let geometries = match &e.geometry {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg: &BlasTriangleGeometry<'_>| {
                        DynContextBlasTriangleGeometry {
                            size: tg.size,
                            vertex_buffer: tg.vertex_buffer.id,

                            index_buffer: tg.index_buffer.map(|index_buffer| index_buffer.id),

                            transform_buffer: tg
                                .transform_buffer
                                .map(|transform_buffer| transform_buffer.id),

                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                            transform_buffer_offset: tg.transform_buffer_offset,
                        }
                    });
                    DynContextBlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            DynContextBlasBuildEntry {
                blas_id: e.blas.id,
                geometries,
            }
        });

        let mut tlas = tlas
            .into_iter()
            .map(|e: &TlasBuildEntry<'_>| DynContextTlasBuildEntry {
                tlas_id: e.tlas.id,
                instance_buffer_id: e.instance_buffer.id,
                instance_count: e.instance_count,
            });

        DynContext::command_encoder_build_acceleration_structures_unsafe_tlas(
            &*self.context,
            id,
            self.data.as_ref(),
            &mut blas,
            &mut tlas,
        );
    }
}
