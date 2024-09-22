use std::{fmt::Debug, ops::Range, sync::Arc, thread};
use wgt::WasmNotSendSync;

use crate::{
    context::{Context, DynContext},
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
pub(crate) struct BlasShared {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
static_assertions::assert_impl_all!(BlasShared: WasmNotSendSync);

#[derive(Debug)]
/// Bottom level acceleration structure or BLAS for short.
/// A BLAS contains geometry in a device readable format, you can't interact directly with this,
/// instead you have to build the BLAS with the buffers containing triangles or AABB.
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

#[derive(Debug)]
/// Top level acceleration structure or TLAS for short.
/// A TLAS contains TLAS instances in a device readable form, you cant interact
/// directly with these, instead you have to build the TLAS with [TLAS instances].
///
/// [TLAS instances]: TlasInstance
pub struct Tlas {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
    pub(crate) max_instances: u32,
}
static_assertions::assert_impl_all!(Tlas: WasmNotSendSync);

impl Tlas {
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::tlas_destroy(&*self.context, self.data.as_ref());
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.tlas_drop(self.data.as_ref());
        }
    }
}

/// Entry for a top level acceleration structure build.
/// Used with raw instance buffers for an unvalidated builds.
/// See [TlasPackage] for the safe version.
pub struct TlasBuildEntry<'a> {
    /// Reference to the acceleration structure.
    pub tlas: &'a Tlas,
    /// Reference to the raw instance buffer, each instance is similar to [TlasInstance] but contains a handle to the BLAS.
    pub instance_buffer: &'a Buffer,
    /// Number of instances in the instance buffer.
    pub instance_count: u32,
}
static_assertions::assert_impl_all!(TlasBuildEntry<'_>: WasmNotSendSync);

/// Safe instance for a [Tlas].
/// A TlasInstance may be made invalid, if a TlasInstance is invalid, any attempt to build a [TlasPackage] containing an
/// invalid TlasInstance will generate a validation error
/// Each one contains:
/// - A reference to a BLAS, this ***must*** be interacted with using [TlasInstance::new] or [TlasInstance::set_blas], a
/// TlasInstance that references a BLAS keeps that BLAS from being dropped, but if the BLAS is explicitly destroyed (e.g.
/// using [Blas::destroy]) the TlasInstance becomes invalid
/// - A user accessible transformation matrix
/// - A user accessible mask
/// - A user accessible custom index
#[derive(Debug, Clone)]
pub struct TlasInstance {
    pub(crate) blas: Arc<BlasShared>,
    /// Affine transform matrix 3x4 (rows x columns, row major order).
    pub transform: [f32; 12],
    /// Custom index for the instance used inside the shader.
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

/// The safe version of TlasEntry, containing TlasInstances instead of a raw buffer.
pub struct TlasPackage {
    pub(crate) tlas: Tlas,
    pub(crate) instances: Vec<Option<TlasInstance>>,
    pub(crate) lowest_unmodified: u32,
}
static_assertions::assert_impl_all!(TlasPackage: WasmNotSendSync);

impl TlasPackage {
    /// Construct [TlasPackage] consuming the [Tlas] (prevents modification of the [Tlas] without using this package).
    pub fn new(tlas: Tlas) -> Self {
        let max_instances = tlas.max_instances;
        Self::new_with_instances(tlas, vec![None; max_instances as usize])
    }

    /// Construct [TlasPackage] consuming the [Tlas] (prevents modification of the Tlas without using this package).
    /// This constructor moves the instances into the package (the number of instances needs to fit into tlas,
    /// otherwise when building a validation error will be raised).
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
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used.
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
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used.
    pub fn get_mut_single(&mut self, index: usize) -> Option<&mut Option<TlasInstance>> {
        if index >= self.instances.len() {
            return None;
        }
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        Some(&mut self.instances[index])
    }

    /// Get the binding resource for the underling acceleration structure, to be used when creating a [BindGroup]
    ///
    /// [BindGroup]: super::BindGroup
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(&self.tlas)
    }

    /// Get a reference to the underling [Tlas].
    pub fn tlas(&self) -> &Tlas {
        &self.tlas
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

pub(crate) struct DynContextTlasBuildEntry<'a> {
    pub(crate) tlas_data: &'a Data,
    pub(crate) instance_buffer_data: &'a Data,
    pub(crate) instance_count: u32,
}

pub(crate) struct DynContextTlasPackage<'a> {
    pub(crate) tlas_data: &'a Data,
    pub(crate) instances: Box<dyn Iterator<Item = Option<DynContextTlasInstance<'a>>> + 'a>,
    pub(crate) lowest_unmodified: u32,
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

/// Context version see [TlasBuildEntry].
#[allow(dead_code)]
pub struct ContextTlasBuildEntry<'a, T: Context> {
    pub(crate) tlas_data: &'a T::TlasData,
    pub(crate) instance_buffer_data: &'a T::BufferData,
    pub(crate) instance_count: u32,
}

/// Context version see [TlasPackage].
#[allow(dead_code)]
pub struct ContextTlasPackage<'a, T: Context> {
    pub(crate) tlas_data: &'a T::TlasData,
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
    /// - `desc`: The descriptor of the acceleration structure.
    /// - `sizes`: Size descriptor limiting what can be built into the acceleration structure.
    ///
    /// # Validation
    /// If any of the following is not satisfied a validation error is generated
    ///
    /// The device ***must*** have [Features::RAY_TRACING_ACCELERATION_STRUCTURE] enabled.
    /// if `sizes` is [BlasGeometrySizeDescriptors::Triangles] then the following must be satisfied
    /// - For every geometry descriptor (for the purposes this is called `geo_desc`) of `sizes.descriptors` the following must be satisfied:
    ///     - `geo_desc.vertex_format` must be within allowed formats (allowed formats for a given feature set
    /// may be queried with [Features::allowed_vertex_formats_for_blas]).
    ///     - Both or neither of `geo_desc.index_format` and `geo_desc.index_count` must be provided.
    ///
    /// [Features::RAY_TRACING_ACCELERATION_STRUCTURE]: wgt::Features::RAY_TRACING_ACCELERATION_STRUCTURE
    /// [Features::allowed_vertex_formats_for_blas]: wgt::Features::allowed_vertex_formats_for_blas
    fn create_blas(
        &self,
        desc: &CreateBlasDescriptor<'_>,
        sizes: BlasGeometrySizeDescriptors,
    ) -> Blas;

    /// Create a top level acceleration structure, used for ray tracing.
    /// - `desc`: The descriptor of the acceleration structure.
    ///
    /// # Validation
    /// If any of the following is not satisfied a validation error is generated
    ///
    /// The device ***must*** have [Features::RAY_TRACING_ACCELERATION_STRUCTURE] enabled.
    ///
    /// [Features::RAY_TRACING_ACCELERATION_STRUCTURE]: wgt::Features::RAY_TRACING_ACCELERATION_STRUCTURE
    fn create_tlas(&self, desc: &CreateTlasDescriptor<'_>) -> Tlas;
}

impl DeviceRayTracing for Device {
    fn create_blas(
        &self,
        desc: &CreateBlasDescriptor<'_>,
        sizes: BlasGeometrySizeDescriptors,
    ) -> Blas {
        let (handle, data) =
            DynContext::device_create_blas(&*self.context, self.data.as_ref(), desc, sizes);

        Blas {
            #[allow(clippy::arc_with_non_send_sync)]
            shared: Arc::new(BlasShared {
                context: Arc::clone(&self.context),
                data,
            }),
            handle,
        }
    }

    fn create_tlas(&self, desc: &CreateTlasDescriptor<'_>) -> Tlas {
        let data = DynContext::device_create_tlas(&*self.context, self.data.as_ref(), desc);

        Tlas {
            context: Arc::clone(&self.context),
            data,
            max_instances: desc.max_instances,
        }
    }
}

/// Trait to add ray tracing functions to a [`CommandEncoder`].
pub trait CommandEncoderRayTracing {
    /// Build bottom and top level acceleration structures.
    /// Builds the BLASes then the TLASes, but does ***not*** build the BLASes into the TLASes,
    /// that must be done by setting a TLAS instance in the TLAS package to one that contains the BLAS (and with an appropriate transform)
    ///
    /// # Validation
    ///
    /// - blas: Iterator of bottom level acceleration structure entries to build.
    ///     For each entry, the provided size descriptor must be strictly smaller or equal to the descriptor given at BLAS creation, this means:
    ///     - Less or equal number of geometries
    ///     - Same kind of geometry (with index buffer or without) (same vertex/index format)
    ///     - Same flags
    ///     - Less or equal number of vertices
    ///     - Less or equal number of indices (if applicable)
    /// - tlas: iterator of top level acceleration structure packages to build
    ///     For each entry:
    ///     - Each BLAS in each TLAS instance must have been being built in the current call or in a previous call to `build_acceleration_structures` or `build_acceleration_structures_unsafe_tlas`
    ///     - The number of TLAS instances must be less than or equal to the max number of tlas instances when creating (if creating a package with `TlasPackage::new()` this is already satisfied)
    ///
    /// If the device the command encoder is created from does not have [Features::RAY_TRACING_ACCELERATION_STRUCTURE] enabled then a validation error is generated
    ///
    /// A bottom level acceleration structure may be build and used as a reference in a top level acceleration structure in the same invocation of this function.
    ///
    /// # Bind group usage
    ///
    /// When a top level acceleration structure is used in a bind group, some validation takes place:
    ///    - The top level acceleration structure is valid and has been built.
    ///    - All the bottom level acceleration structures referenced by the top level acceleration structure are valid and have been built prior,
    ///      or at same time as the containing top level acceleration structure.
    ///
    /// [Features::RAY_TRACING_ACCELERATION_STRUCTURE]: wgt::Features::RAY_TRACING_ACCELERATION_STRUCTURE
    fn build_acceleration_structures<'a>(
        &mut self,
        blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>,
        tlas: impl IntoIterator<Item = &'a TlasPackage>,
    );

    /// Build bottom and top level acceleration structures.
    /// See [`CommandEncoderRayTracing::build_acceleration_structures`] for the safe version and more details. All validation in [`CommandEncoderRayTracing::build_acceleration_structures`] except that
    /// listed under tlas applies here as well.
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
        let mut blas = blas.into_iter().map(|e: &BlasBuildEntry<'_>| {
            let geometries = match &e.geometry {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries
                        .iter()
                        .map(
                            |tg: &BlasTriangleGeometry<'_>| DynContextBlasTriangleGeometry {
                                size: tg.size,
                                vertex_buffer: tg.vertex_buffer.data.as_ref(),

                                index_buffer: tg
                                    .index_buffer
                                    .map(|index_buffer| index_buffer.data.as_ref()),

                                transform_buffer: tg
                                    .transform_buffer
                                    .map(|transform_buffer| transform_buffer.data.as_ref()),

                                first_vertex: tg.first_vertex,
                                vertex_stride: tg.vertex_stride,
                                index_buffer_offset: tg.index_buffer_offset,
                                transform_buffer_offset: tg.transform_buffer_offset,
                            },
                        );
                    DynContextBlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            DynContextBlasBuildEntry {
                blas_data: e.blas.shared.data.as_ref(),
                geometries,
            }
        });

        let mut tlas = tlas.into_iter().map(|e: &TlasPackage| {
            let instances = e.instances.iter().map(|instance: &Option<TlasInstance>| {
                instance.as_ref().map(|instance| DynContextTlasInstance {
                    blas: instance.blas.data.as_ref(),
                    transform: &instance.transform,
                    custom_index: instance.custom_index,
                    mask: instance.mask,
                })
            });
            DynContextTlasPackage {
                tlas_data: e.tlas.data.as_ref(),
                instances: Box::new(instances),
                lowest_unmodified: e.lowest_unmodified,
            }
        });

        DynContext::command_encoder_build_acceleration_structures(
            &*self.context,
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
        let mut blas = blas.into_iter().map(|e: &BlasBuildEntry<'_>| {
            let geometries = match &e.geometry {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries
                        .iter()
                        .map(
                            |tg: &BlasTriangleGeometry<'_>| DynContextBlasTriangleGeometry {
                                size: tg.size,
                                vertex_buffer: tg.vertex_buffer.data.as_ref(),

                                index_buffer: tg
                                    .index_buffer
                                    .map(|index_buffer| index_buffer.data.as_ref()),

                                transform_buffer: tg
                                    .transform_buffer
                                    .map(|transform_buffer| transform_buffer.data.as_ref()),

                                first_vertex: tg.first_vertex,
                                vertex_stride: tg.vertex_stride,
                                index_buffer_offset: tg.index_buffer_offset,
                                transform_buffer_offset: tg.transform_buffer_offset,
                            },
                        );
                    DynContextBlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            DynContextBlasBuildEntry {
                blas_data: e.blas.shared.data.as_ref(),
                geometries,
            }
        });

        let mut tlas = tlas
            .into_iter()
            .map(|e: &TlasBuildEntry<'_>| DynContextTlasBuildEntry {
                tlas_data: e.tlas.data.as_ref(),
                instance_buffer_data: e.instance_buffer.data.as_ref(),
                instance_count: e.instance_count,
            });

        DynContext::command_encoder_build_acceleration_structures_unsafe_tlas(
            &*self.context,
            self.data.as_ref(),
            &mut blas,
            &mut tlas,
        );
    }
}