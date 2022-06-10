use crate::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, RenderBundle},
    device::Device,
    id,
    instance::{Adapter, HalSurface, Instance, Surface},
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    registry,
    resource::{Buffer, QuerySet, Sampler, Texture, TextureClearMode, TextureView},
    Epoch, Index,
};

use parking_lot::Mutex;
use thiserror::Error;
use wgt::Backend;

use std::borrow::Cow;
#[cfg(debug_assertions)]
use std::{fmt::Debug, marker::PhantomData};

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`free`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// Use `IdentityManager::default` to construct new instances.
///
/// `IdentityManager` returns `Id`s whose index values are suitable for use as
/// indices into a `Storage<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Each live id's index selects a
///   distinct element in the vector.
///
/// - `IdentityManager` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityManager` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// [`Id`]: crate::id::Id
/// [`Backend`]: wgt::Backend;
/// [`alloc`]: IdentityManager::alloc
/// [`free`]: IdentityManager::free
#[derive(Debug, Default)]
pub struct IdentityManager {
    /// Available index values. If empty, then `epochs.len()` is the next index
    /// to allocate.
    free: Vec<Index>,

    /// The next or currently-live epoch value associated with each `Id` index.
    ///
    /// If there is a live id with index `i`, then `epochs[i]` is its epoch; any
    /// id with the same index but an older epoch is dead.
    ///
    /// If index `i` is currently unused, `epochs[i]` is the epoch to use in its
    /// next `Id`.
    epochs: Vec<Epoch>,
}

impl IdentityManager {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<I: id::TypedId>(&mut self, backend: Backend) -> (Index, I) {
        match self.free.pop() {
            Some(index) => {
                let id = I::zip(index, self.epochs[index as usize], backend);
                (index, id)
            }
            None => {
                let epoch = 1;
                let index = self.epochs.len() as Index;
                let id = I::zip(index, epoch, backend);
                self.epochs.push(epoch);
                (index, id)
            }
        }
    }

    /// Free `id`. It will never be returned from `alloc` again.
    pub fn free<I: id::TypedId>(&mut self, id: I) -> (Index, Epoch) {
        let (index, epoch, _backend) = id.unzip();
        let pe = &mut self.epochs[index as usize];
        assert_eq!(*pe, epoch);
        // If the epoch reaches EOL, the index doesn't go
        // into the free list, will never be reused again.
        if epoch < id::EPOCH_MASK {
            *pe = epoch + 1;
            self.free.push(index);
        }
        (index, epoch)
    }
}

/// An entry in a `Storage::map` table.
#[derive(Debug)]
enum Element<T> {
    /// There are no live ids with this index.
    Vacant,

    /// There is one live id with this index, allocated at the given
    /// epoch.
    Occupied(T, Epoch),

    /// Like `Occupied`, but an error occurred when creating the
    /// resource.
    ///
    /// The given `String` is the resource's descriptor label.
    Error(Epoch, String),
}

#[derive(Clone, Debug, Default)]
pub struct StorageReport {
    pub num_occupied: usize,
    pub num_vacant: usize,
    pub num_error: usize,
    pub element_size: usize,
}

impl StorageReport {
    pub fn is_empty(&self) -> bool {
        self.num_occupied + self.num_vacant + self.num_error == 0
    }
}

#[derive(Clone, Debug, Error)]
pub(crate) enum InvalidId {
    #[error("Resource {index} is in the error state because {error}. This happens when resource creation fails with an error and the ID is re-used.")]
    ResourceInError { index: u32, error: Cow<'static, str> },
    #[error("Resource {index} has been destroyed. The storage is currently empty.")]
    Vacant { index: u32 },
    #[error("Resource {index} has been destroyed. The storage is currently storing a new resource with epoch {new} (given epoch {old}).")]
    WrongEpoch { index: u32, old: Epoch, new: Epoch },
}

pub trait IdentityHandler<I>: Debug {
    type Input: Clone + Debug;
    fn process(&self, id: Self::Input, backend: Backend) -> I;
    fn free(&self, id: I);
}

impl<I: id::TypedId + Debug> IdentityHandler<I> for Mutex<IdentityManager> {
    type Input = PhantomData<I>;
    fn process(&self, _id: Self::Input, backend: Backend) -> I {
        self.lock().alloc(backend).1
    }
    fn free(&self, id: I) {
        self.lock().free(id);
    }
}

pub trait IdentityHandlerFactory<I> {
    type Filter: IdentityHandler<I>;
    fn spawn(&self) -> Self::Filter;
}

#[derive(Debug)]
pub struct IdentityManagerFactory;

impl<I: id::TypedId + Debug> IdentityHandlerFactory<I> for IdentityManagerFactory {
    type Filter = Mutex<IdentityManager>;
    fn spawn(&self) -> Self::Filter {
        Mutex::new(IdentityManager::default())
    }
}

pub trait GlobalIdentityHandlerFactory:
    IdentityHandlerFactory<id::AdapterId>
    + IdentityHandlerFactory<id::DeviceId>
    + IdentityHandlerFactory<id::PipelineLayoutId>
    + IdentityHandlerFactory<id::ShaderModuleId>
    + IdentityHandlerFactory<id::BindGroupLayoutId>
    + IdentityHandlerFactory<id::BindGroupId>
    + IdentityHandlerFactory<id::CommandBufferId>
    + IdentityHandlerFactory<id::RenderBundleId>
    + IdentityHandlerFactory<id::RenderPipelineId>
    + IdentityHandlerFactory<id::ComputePipelineId>
    + IdentityHandlerFactory<id::QuerySetId>
    + IdentityHandlerFactory<id::BufferId>
    + IdentityHandlerFactory<id::TextureId>
    + IdentityHandlerFactory<id::TextureViewId>
    + IdentityHandlerFactory<id::SamplerId>
    + IdentityHandlerFactory<id::SurfaceId>
{
}

impl GlobalIdentityHandlerFactory for IdentityManagerFactory {}

pub type Input<G, I> = <<G as IdentityHandlerFactory<I>>::Filter as IdentityHandler<I>>::Input;

pub trait Resource {
    type Id: id::TypedId;
    const TYPE: &'static str;
    fn life_guard(&self) -> Option<&crate::LifeGuard>;
    fn device_id(&self) -> id::Valid<id::DeviceId>;
    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.life_guard().unwrap().label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

#[must_use]
pub(crate) struct FutureId<'a, T: Resource> {
    id: T::Id,
    data: &'a registry::Storage<T>,
}

impl<T> FutureId<'_, T>
where
    T: Resource,
{
    #[cfg(feature = "trace")]
    pub fn id(&self) -> T::Id {
        self.id
    }

    pub fn into_id(self) -> T::Id {
        self.id
    }

    pub fn assign(self, value: T) -> id::Valid<T::Id> {
        use id::TypedId as _;
        let (index, epoch, _) = self.id.unzip();
        unsafe { self.data.fill(index, epoch, Ok(value)) };
        id::Valid(self.id)
    }

    pub fn assign_error(self, label: &str) -> T::Id {
        use id::TypedId as _;
        let (index, epoch, _) = self.id.unzip();
        unsafe {
            self.data
                .fill(index, epoch, Err(Cow::Owned(String::from(label))))
        };
        self.id
    }
}

#[derive(Debug)]
pub struct HubReport {
    pub adapters: StorageReport,
    pub devices: StorageReport,
    pub pipeline_layouts: StorageReport,
    pub shader_modules: StorageReport,
    pub bind_group_layouts: StorageReport,
    pub bind_groups: StorageReport,
    pub command_buffers: StorageReport,
    pub render_bundles: StorageReport,
    pub render_pipelines: StorageReport,
    pub compute_pipelines: StorageReport,
    pub query_sets: StorageReport,
    pub buffers: StorageReport,
    pub textures: StorageReport,
    pub texture_views: StorageReport,
    pub samplers: StorageReport,
}

impl HubReport {
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

pub struct Hub<A: HalApi, F: GlobalIdentityHandlerFactory> {
    pub adapters: registry::Registry<A, Adapter<A>>,
    pub devices: registry::Registry<A, Device<A>>,
    pub pipeline_layouts: registry::Registry<A, PipelineLayout<A>>,
    pub shader_modules: registry::Registry<A, ShaderModule<A>>,
    pub bind_group_layouts: registry::Registry<A, BindGroupLayout<A>>,
    pub bind_groups: registry::Registry<A, BindGroup<A>>,
    pub command_buffers: registry::Registry<A, CommandBuffer<A>>,
    pub render_bundles: registry::Registry<A, RenderBundle<A>>,
    pub render_pipelines: registry::Registry<A, RenderPipeline<A>>,
    pub compute_pipelines: registry::Registry<A, ComputePipeline<A>>,
    pub query_sets: registry::Registry<A, QuerySet<A>>,
    pub buffers: registry::Registry<A, Buffer<A>>,
    pub textures: registry::Registry<A, Texture<A>>,
    pub texture_views: registry::Registry<A, TextureView<A>>,
    pub samplers: registry::Registry<A, Sampler<A>>,
    _phantom: PhantomData<F>,
}

impl<A: HalApi, F: GlobalIdentityHandlerFactory> Hub<A, F> {
    fn new(_factory: &F) -> Self {
        Self {
            adapters: registry::Registry::new(),
            devices: registry::Registry::new(),
            pipeline_layouts: registry::Registry::new(),
            shader_modules: registry::Registry::new(),
            bind_group_layouts: registry::Registry::new(),
            bind_groups: registry::Registry::new(),
            command_buffers: registry::Registry::new(),
            render_bundles: registry::Registry::new(),
            render_pipelines: registry::Registry::new(),
            compute_pipelines: registry::Registry::new(),
            query_sets: registry::Registry::new(),
            buffers: registry::Registry::new(),
            textures: registry::Registry::new(),
            texture_views: registry::Registry::new(),
            samplers: registry::Registry::new(),
            _phantom: PhantomData,
        }
    }

    //TODO: instead of having a hacky `with_adapters` parameter,
    // we should have `clear_device(device_id)` that specifically destroys
    // everything related to a logical device.
    unsafe fn clear(
        &self,
        surface_guard: &registry::Registry<hal::api::Empty, Surface>,
        with_adapters: bool,
    ) {
        use crate::resource::TextureInner;
        use hal::{Device as _, Surface as _};

        for (_, device) in self.devices.iter_mut() {
            device.prepare_to_die();
        }

        // destroy command buffers first, since otherwise DX12 isn't happy
        for command_buffer in self.command_buffers.remove_all() {
            let device = &self.devices[command_buffer.device_id.value];
            device.destroy_command_buffer(command_buffer);
        }

        for sampler in self.samplers.remove_all() {
            unsafe {
                self.devices[sampler.device_id.value]
                    .raw
                    .destroy_sampler(sampler.raw);
            }
        }

        for texture_view in self.texture_views.remove_all() {
            let device = &self.devices[texture_view.device_id.value];
            unsafe {
                device.raw.destroy_texture_view(texture_view.raw);
            }
        }

        for texture in self.textures.remove_all() {
            let device = &self.devices[texture.device_id.value];
            if let TextureInner::Native { raw: Some(raw) } = texture.inner {
                unsafe {
                    device.raw.destroy_texture(raw);
                }
            }
            if let TextureClearMode::RenderPass { clear_views, .. } = texture.clear_mode {
                for view in clear_views {
                    unsafe {
                        device.raw.destroy_texture_view(view);
                    }
                }
            }
        }
        for buffer in self.buffers.remove_all() {
            //TODO: unmap if needed
            self.devices[buffer.device_id.value].destroy_buffer(buffer);
        }
        for bind_group in self.bind_groups.remove_all() {
            let device = &self.devices[bind_group.device_id.value];
            unsafe {
                device.raw.destroy_bind_group(bind_group.raw);
            }
        }

        for module in self.shader_modules.remove_all() {
            let device = &self.devices[module.device_id.value];
            unsafe {
                device.raw.destroy_shader_module(module.raw);
            }
        }
        for bgl in self.bind_group_layouts.remove_all() {
            let device = &self.devices[bgl.device_id.value];
            unsafe {
                device.raw.destroy_bind_group_layout(bgl.raw);
            }
        }
        for pipeline_layout in self.pipeline_layouts.remove_all() {
            let device = &self.devices[pipeline_layout.device_id.value];
            unsafe {
                device.raw.destroy_pipeline_layout(pipeline_layout.raw);
            }
        }
        for pipeline in self.compute_pipelines.remove_all() {
            let device = &self.devices[pipeline.device_id.value];
            unsafe {
                device.raw.destroy_compute_pipeline(pipeline.raw);
            }
        }
        for pipeline in self.render_pipelines.remove_all() {
            let device = &self.devices[pipeline.device_id.value];
            unsafe {
                device.raw.destroy_render_pipeline(pipeline.raw);
            }
        }

        for (_, surface) in surface_guard.iter_mut() {
            if surface
                .presentation
                .as_ref()
                .map_or(wgt::Backend::Empty, |p| p.backend())
                != A::VARIANT
            {
                continue;
            }
            if let Some(present) = surface.presentation.take() {
                let device = &self.devices[present.device_id.value];
                let suf = A::get_surface(surface);
                unsafe {
                    suf.raw.lock().unconfigure(&device.raw);
                    //TODO: we could destroy the surface here
                }
            }
        }

        for query_set in self.query_sets.remove_all() {
            let device = &self.devices[query_set.device_id.value];
            unsafe {
                device.raw.destroy_query_set(query_set.raw);
            }
        }

        for device in self.devices.remove_all() {
            device.dispose();
        }

        if with_adapters {
            self.adapters.remove_all();
        }
    }

    pub fn generate_report(&self) -> HubReport {
        HubReport {
            adapters: self.adapters.generate_report(),
            devices: self.devices.generate_report(),
            pipeline_layouts: self.pipeline_layouts.generate_report(),
            shader_modules: self.shader_modules.generate_report(),
            bind_group_layouts: self.bind_group_layouts.generate_report(),
            bind_groups: self.bind_groups.generate_report(),
            command_buffers: self.command_buffers.generate_report(),
            render_bundles: self.render_bundles.generate_report(),
            render_pipelines: self.render_pipelines.generate_report(),
            compute_pipelines: self.compute_pipelines.generate_report(),
            query_sets: self.query_sets.generate_report(),
            buffers: self.buffers.generate_report(),
            textures: self.textures.generate_report(),
            texture_views: self.texture_views.generate_report(),
            samplers: self.samplers.generate_report(),
        }
    }
}

pub struct Hubs<F: GlobalIdentityHandlerFactory> {
    #[cfg(vulkan)]
    vulkan: Hub<hal::api::Vulkan, F>,
    #[cfg(metal)]
    metal: Hub<hal::api::Metal, F>,
    #[cfg(dx12)]
    dx12: Hub<hal::api::Dx12, F>,
    #[cfg(dx11)]
    dx11: Hub<hal::api::Dx11, F>,
    #[cfg(gl)]
    gl: Hub<hal::api::Gles, F>,
}

impl<F: GlobalIdentityHandlerFactory> Hubs<F> {
    fn new(factory: &F) -> Self {
        Self {
            #[cfg(vulkan)]
            vulkan: Hub::new(factory),
            #[cfg(metal)]
            metal: Hub::new(factory),
            #[cfg(dx12)]
            dx12: Hub::new(factory),
            #[cfg(dx11)]
            dx11: Hub::new(factory),
            #[cfg(gl)]
            gl: Hub::new(factory),
        }
    }
}

#[derive(Debug)]
pub struct GlobalReport {
    pub surfaces: StorageReport,
    #[cfg(vulkan)]
    pub vulkan: Option<HubReport>,
    #[cfg(metal)]
    pub metal: Option<HubReport>,
    #[cfg(dx12)]
    pub dx12: Option<HubReport>,
    #[cfg(dx11)]
    pub dx11: Option<HubReport>,
    #[cfg(gl)]
    pub gl: Option<HubReport>,
}

pub struct Global<G: GlobalIdentityHandlerFactory> {
    pub instance: Instance,
    pub surfaces: registry::Registry<hal::api::Empty, Surface>,
    hubs: Hubs<G>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn new(name: &str, factory: G, backends: wgt::Backends) -> Self {
        profiling::scope!("new", "Global");
        Self {
            instance: Instance::new(name, backends),
            surfaces: registry::Registry::new(),
            hubs: Hubs::new(&factory),
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: HalApi>(
        name: &str,
        factory: G,
        hal_instance: A::Instance,
    ) -> Self {
        profiling::scope!("new", "Global");
        Self {
            instance: A::create_instance_from_hal(name, hal_instance),
            surfaces: registry::Registry::new(),
            hubs: Hubs::new(&factory),
        }
    }

    /// # Safety
    ///
    /// - The raw handle obtained from the hal Instance must not be manually destroyed
    pub unsafe fn instance_as_hal<A: HalApi, F: FnOnce(Option<&A::Instance>) -> R, R>(
        &self,
        hal_instance_callback: F,
    ) -> R {
        let hal_instance = A::instance_as_hal(&self.instance);
        hal_instance_callback(hal_instance)
    }

    pub unsafe fn clear_backend<A: HalApi>(&self, _dummy: ()) {
        let hub = A::hub(self);
        // this is used for tests, which keep the adapter
        unsafe { hub.clear(&self.surfaces, false) };
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            #[cfg(vulkan)]
            vulkan: if self.instance.vulkan.is_some() {
                Some(self.hubs.vulkan.generate_report())
            } else {
                None
            },
            #[cfg(metal)]
            metal: if self.instance.metal.is_some() {
                Some(self.hubs.metal.generate_report())
            } else {
                None
            },
            #[cfg(dx12)]
            dx12: if self.instance.dx12.is_some() {
                Some(self.hubs.dx12.generate_report())
            } else {
                None
            },
            #[cfg(dx11)]
            dx11: if self.instance.dx11.is_some() {
                Some(self.hubs.dx11.generate_report())
            } else {
                None
            },
            #[cfg(gl)]
            gl: if self.instance.gl.is_some() {
                Some(self.hubs.gl.generate_report())
            } else {
                None
            },
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Drop for Global<G> {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        log::info!("Dropping Global");

        // destroy hubs before the instance gets dropped
        #[cfg(vulkan)]
        unsafe {
            self.hubs.vulkan.clear(&self.surfaces, true);
        }
        #[cfg(metal)]
        unsafe {
            self.hubs.metal.clear(&self.surfaces, true);
        }
        #[cfg(dx12)]
        unsafe {
            self.hubs.dx12.clear(&self.surfaces, true);
        }
        #[cfg(dx11)]
        unsafe {
            self.hubs.dx11.clear(&self.surfaces, true);
        }
        #[cfg(gl)]
        unsafe {
            self.hubs.gl.clear(&self.surfaces, true);
        }

        // destroy surfaces
        unsafe {
            for surface in self.surfaces.remove_all() {
                self.instance.destroy_surface(surface);
            }
        }
    }
}

pub trait HalApi: hal::Api + 'static {
    const VARIANT: Backend;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance;
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance>;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G>;
    fn get_surface(surface: &Surface) -> &HalSurface<Self>;
}

impl HalApi for hal::api::Empty {
    const VARIANT: Backend = Backend::Empty;
    fn create_instance_from_hal(_: &str, _: Self::Instance) -> Instance {
        unimplemented!("called empty api")
    }
    fn instance_as_hal(_: &Instance) -> Option<&Self::Instance> {
        unimplemented!("called empty api")
    }
    fn hub<G: GlobalIdentityHandlerFactory>(_: &Global<G>) -> &Hub<Self, G> {
        unimplemented!("called empty api")
    }
    fn get_surface(_: &Surface) -> &HalSurface<Self> {
        unimplemented!("called empty api")
    }
}

#[cfg(vulkan)]
impl HalApi for hal::api::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            vulkan: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.vulkan.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.vulkan
    }
    fn get_surface(surface: &Surface) -> &HalSurface<Self> {
        surface.vulkan.as_ref().unwrap()
    }
}

#[cfg(metal)]
impl HalApi for hal::api::Metal {
    const VARIANT: Backend = Backend::Metal;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            metal: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.metal.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.metal
    }
    fn get_surface(surface: &Surface) -> &HalSurface<Self> {
        surface.metal.as_ref().unwrap()
    }
}

#[cfg(dx12)]
impl HalApi for hal::api::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            dx12: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.dx12.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.dx12
    }
    fn get_surface(surface: &Surface) -> &HalSurface<Self> {
        surface.dx12.as_ref().unwrap()
    }
}

#[cfg(dx11)]
impl HalApi for hal::api::Dx11 {
    const VARIANT: Backend = Backend::Dx11;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            dx11: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.dx11.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.dx11
    }
    fn get_surface(surface: &Surface) -> &HalSurface<Self> {
        surface.dx11.as_ref().unwrap()
    }
}

#[cfg(gl)]
impl HalApi for hal::api::Gles {
    const VARIANT: Backend = Backend::Gl;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        #[allow(clippy::needless_update)]
        Instance {
            name: name.to_owned(),
            gl: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.gl.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.gl
    }
    fn get_surface(surface: &Surface) -> &HalSurface<Self> {
        surface.gl.as_ref().unwrap()
    }
}

#[cfg(test)]
fn _test_send_sync(global: &Global<IdentityManagerFactory>) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}

#[test]
fn test_epoch_end_of_life() {
    let mut man = IdentityManager::default();
    man.epochs.push(id::EPOCH_MASK);
    man.free.push(0);
    let (index1, id1) = man.alloc::<id::BufferId>(Backend::Empty);
    assert_eq!(index1, 0);
    man.free(id1);
    let (index2, _) = man.alloc::<id::BufferId>(Backend::Empty);
    // confirm that the index 0 is no longer re-used
    assert_eq!(index2, 1);
}
