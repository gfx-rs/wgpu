/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend,
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, RenderBundle},
    device::Device,
    id::{
        AdapterId, BindGroupId, BindGroupLayoutId, BufferId, CommandBufferId, ComputePipelineId,
        DeviceId, PipelineLayoutId, RenderBundleId, RenderPipelineId, SamplerId, ShaderModuleId,
        SurfaceId, SwapChainId, TextureId, TextureViewId, TypedId, Valid,
    },
    instance::{Adapter, Instance, Surface},
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    resource::{Buffer, Sampler, Texture, TextureView},
    swap_chain::SwapChain,
    Epoch, Index,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use wgt::Backend;

use crate::id::QuerySetId;
use crate::resource::QuerySet;
#[cfg(debug_assertions)]
use std::cell::Cell;
use std::{fmt::Debug, marker::PhantomData, ops, thread};

/// A simple structure to manage identities of objects.
#[derive(Debug)]
pub struct IdentityManager {
    free: Vec<Index>,
    epochs: Vec<Epoch>,
}

impl Default for IdentityManager {
    fn default() -> Self {
        Self {
            free: Default::default(),
            epochs: Default::default(),
        }
    }
}

impl IdentityManager {
    pub fn from_index(min_index: u32) -> Self {
        Self {
            free: (0..min_index).collect(),
            epochs: vec![1; min_index as usize],
        }
    }

    pub fn alloc<I: TypedId>(&mut self, backend: Backend) -> I {
        match self.free.pop() {
            Some(index) => I::zip(index, self.epochs[index as usize], backend),
            None => {
                let epoch = 1;
                let id = I::zip(self.epochs.len() as Index, epoch, backend);
                self.epochs.push(epoch);
                id
            }
        }
    }

    pub fn free<I: TypedId + Debug>(&mut self, id: I) {
        let (index, epoch, _backend) = id.unzip();
        // avoid doing this check in release
        if cfg!(debug_assertions) {
            assert!(!self.free.contains(&index));
        }
        let pe = &mut self.epochs[index as usize];
        assert_eq!(*pe, epoch);
        *pe += 1;
        self.free.push(index);
    }
}

#[derive(Debug)]
enum Element<T> {
    Vacant,
    Occupied(T, Epoch),
    Error(Epoch, String),
}

#[derive(Clone, Debug)]
pub(crate) struct InvalidId;

#[derive(Debug)]
pub struct Storage<T, I: TypedId> {
    map: Vec<Element<T>>,
    kind: &'static str,
    _phantom: PhantomData<I>,
}

impl<T, I: TypedId> ops::Index<Valid<I>> for Storage<T, I> {
    type Output = T;
    fn index(&self, id: Valid<I>) -> &T {
        self.get(id.0).unwrap()
    }
}

impl<T, I: TypedId> ops::IndexMut<Valid<I>> for Storage<T, I> {
    fn index_mut(&mut self, id: Valid<I>) -> &mut T {
        self.get_mut(id.0).unwrap()
    }
}

impl<T, I: TypedId> Storage<T, I> {
    pub(crate) fn contains(&self, id: I) -> bool {
        let (index, epoch, _) = id.unzip();
        match self.map[index as usize] {
            Element::Vacant => false,
            Element::Occupied(_, storage_epoch) | Element::Error(storage_epoch, ..) => {
                epoch == storage_epoch
            }
        }
    }

    /// Get a reference to an item behind a potentially invalid ID.
    /// Panics if there is an epoch mismatch, or the entry is empty.
    pub(crate) fn get(&self, id: I) -> Result<&T, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map[index as usize] {
            Element::Occupied(ref v, epoch) => (Ok(v), epoch),
            Element::Vacant => panic!("{}[{}] does not exist", self.kind, index),
            Element::Error(epoch, ..) => (Err(InvalidId), epoch),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{}] is no longer alive",
            self.kind, index
        );
        result
    }

    /// Get a mutable reference to an item behind a potentially invalid ID.
    /// Panics if there is an epoch mismatch, or the entry is empty.
    pub(crate) fn get_mut(&mut self, id: I) -> Result<&mut T, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map[index as usize] {
            Element::Occupied(ref mut v, epoch) => (Ok(v), epoch),
            Element::Vacant => panic!("{}[{}] does not exist", self.kind, index),
            Element::Error(epoch, ..) => (Err(InvalidId), epoch),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{}] is no longer alive",
            self.kind, index
        );
        result
    }

    pub(crate) fn label_for_invalid_id(&self, id: I) -> &str {
        let (index, _, _) = id.unzip();
        match self.map[index as usize] {
            Element::Error(_, ref label) => label,
            _ => "",
        }
    }

    fn insert_impl(&mut self, index: usize, element: Element<T>) {
        if index >= self.map.len() {
            self.map.resize_with(index + 1, || Element::Vacant);
        }
        match std::mem::replace(&mut self.map[index], element) {
            Element::Vacant => {}
            _ => panic!("Index {:?} is already occupied", index),
        }
    }

    pub(crate) fn insert(&mut self, id: I, value: T) {
        let (index, epoch, _) = id.unzip();
        self.insert_impl(index as usize, Element::Occupied(value, epoch))
    }

    pub(crate) fn insert_error(&mut self, id: I, label: &str) {
        let (index, epoch, _) = id.unzip();
        self.insert_impl(index as usize, Element::Error(epoch, label.to_string()))
    }

    pub(crate) fn remove(&mut self, id: I) -> Option<T> {
        let (index, epoch, _) = id.unzip();
        match std::mem::replace(&mut self.map[index as usize], Element::Vacant) {
            Element::Occupied(value, storage_epoch) => {
                assert_eq!(epoch, storage_epoch);
                Some(value)
            }
            Element::Error(..) => None,
            Element::Vacant => panic!("Cannot remove a vacant resource"),
        }
    }

    // Prevents panic on out of range access, allows Vacant elements.
    pub(crate) fn try_remove(&mut self, id: I) -> Option<T> {
        let (index, epoch, _) = id.unzip();
        if index as usize >= self.map.len() {
            None
        } else if let Element::Occupied(value, storage_epoch) =
            std::mem::replace(&mut self.map[index as usize], Element::Vacant)
        {
            assert_eq!(epoch, storage_epoch);
            Some(value)
        } else {
            None
        }
    }

    pub(crate) fn iter(&self, backend: Backend) -> impl Iterator<Item = (I, &T)> {
        self.map
            .iter()
            .enumerate()
            .filter_map(move |(index, x)| match *x {
                Element::Occupied(ref value, storage_epoch) => {
                    Some((I::zip(index as Index, storage_epoch, backend), value))
                }
                _ => None,
            })
    }
}

/// Type system for enforcing the lock order on shared HUB structures.
/// If type A implements `Access<B>`, that means we are allowed to proceed
/// with locking resource `B` after we lock `A`.
///
/// The implenentations basically describe the edges in a directed graph
/// of lock transitions. As long as it doesn't have loops, we can have
/// multiple concurrent paths on this graph (from multiple threads) without
/// deadlocks, i.e. there is always a path whose next resource is not locked
/// by some other path, at any time.
pub trait Access<B> {}

pub enum Root {}
//TODO: establish an order instead of declaring all the pairs.
impl Access<Instance> for Root {}
impl Access<Surface> for Root {}
impl Access<Surface> for Instance {}
impl<B: hal::Backend> Access<Adapter<B>> for Root {}
impl<B: hal::Backend> Access<Adapter<B>> for Surface {}
impl<B: hal::Backend> Access<Device<B>> for Root {}
impl<B: hal::Backend> Access<Device<B>> for Surface {}
impl<B: hal::Backend> Access<Device<B>> for Adapter<B> {}
impl<B: hal::Backend> Access<SwapChain<B>> for Root {}
impl<B: hal::Backend> Access<SwapChain<B>> for Device<B> {}
impl<B: hal::Backend> Access<PipelineLayout<B>> for Root {}
impl<B: hal::Backend> Access<PipelineLayout<B>> for Device<B> {}
impl<B: hal::Backend> Access<PipelineLayout<B>> for RenderBundle {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Root {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for Root {}
impl<B: hal::Backend> Access<BindGroup<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Root {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for SwapChain<B> {}
impl<B: hal::Backend> Access<RenderBundle> for Device<B> {}
impl<B: hal::Backend> Access<RenderBundle> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for Device<B> {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for Device<B> {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for ComputePipeline<B> {}
impl<B: hal::Backend> Access<QuerySet<B>> for Root {}
impl<B: hal::Backend> Access<QuerySet<B>> for Device<B> {}
impl<B: hal::Backend> Access<QuerySet<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<QuerySet<B>> for RenderPipeline<B> {}
impl<B: hal::Backend> Access<QuerySet<B>> for ComputePipeline<B> {}
impl<B: hal::Backend> Access<ShaderModule<B>> for Device<B> {}
impl<B: hal::Backend> Access<ShaderModule<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for Root {}
impl<B: hal::Backend> Access<Buffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for ComputePipeline<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for RenderPipeline<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for QuerySet<B> {}
impl<B: hal::Backend> Access<Texture<B>> for Root {}
impl<B: hal::Backend> Access<Texture<B>> for Device<B> {}
impl<B: hal::Backend> Access<Texture<B>> for Buffer<B> {}
impl<B: hal::Backend> Access<TextureView<B>> for Root {}
impl<B: hal::Backend> Access<TextureView<B>> for SwapChain<B> {}
impl<B: hal::Backend> Access<TextureView<B>> for Device<B> {}
impl<B: hal::Backend> Access<TextureView<B>> for Texture<B> {}
impl<B: hal::Backend> Access<Sampler<B>> for Root {}
impl<B: hal::Backend> Access<Sampler<B>> for Device<B> {}
impl<B: hal::Backend> Access<Sampler<B>> for TextureView<B> {}

#[cfg(debug_assertions)]
thread_local! {
    static ACTIVE_TOKEN: Cell<u8> = Cell::new(0);
}

/// A permission token to lock resource `T` or anything after it,
/// as defined by the `Access` implementations.
///
/// Note: there can only be one non-borrowed `Token` alive on a thread
/// at a time, which is enforced by `ACTIVE_TOKEN`.
pub(crate) struct Token<'a, T: 'a> {
    level: PhantomData<&'a T>,
}

impl<'a, T> Token<'a, T> {
    fn new() -> Self {
        #[cfg(debug_assertions)]
        ACTIVE_TOKEN.with(|active| {
            let old = active.get();
            assert_ne!(old, 0, "Root token was dropped");
            active.set(old + 1);
        });
        Self { level: PhantomData }
    }
}

impl Token<'static, Root> {
    pub fn root() -> Self {
        #[cfg(debug_assertions)]
        ACTIVE_TOKEN.with(|active| {
            assert_eq!(0, active.replace(1), "Root token is already active");
        });

        Self { level: PhantomData }
    }
}

impl<'a, T> Drop for Token<'a, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        ACTIVE_TOKEN.with(|active| {
            let old = active.get();
            active.set(old - 1);
        });
    }
}

pub trait IdentityHandler<I>: Debug {
    type Input: Clone + Debug;
    fn process(&self, id: Self::Input, backend: Backend) -> I;
    fn free(&self, id: I);
}

impl<I: TypedId + Debug> IdentityHandler<I> for Mutex<IdentityManager> {
    type Input = PhantomData<I>;
    fn process(&self, _id: Self::Input, backend: Backend) -> I {
        self.lock().alloc(backend)
    }
    fn free(&self, id: I) {
        self.lock().free(id)
    }
}

pub trait IdentityHandlerFactory<I> {
    type Filter: IdentityHandler<I>;
    fn spawn(&self, min_index: Index) -> Self::Filter;
}

#[derive(Debug)]
pub struct IdentityManagerFactory;

impl<I: TypedId + Debug> IdentityHandlerFactory<I> for IdentityManagerFactory {
    type Filter = Mutex<IdentityManager>;
    fn spawn(&self, min_index: Index) -> Self::Filter {
        Mutex::new(IdentityManager::from_index(min_index))
    }
}

pub trait GlobalIdentityHandlerFactory:
    IdentityHandlerFactory<AdapterId>
    + IdentityHandlerFactory<DeviceId>
    + IdentityHandlerFactory<SwapChainId>
    + IdentityHandlerFactory<PipelineLayoutId>
    + IdentityHandlerFactory<ShaderModuleId>
    + IdentityHandlerFactory<BindGroupLayoutId>
    + IdentityHandlerFactory<BindGroupId>
    + IdentityHandlerFactory<CommandBufferId>
    + IdentityHandlerFactory<RenderBundleId>
    + IdentityHandlerFactory<RenderPipelineId>
    + IdentityHandlerFactory<ComputePipelineId>
    + IdentityHandlerFactory<QuerySetId>
    + IdentityHandlerFactory<BufferId>
    + IdentityHandlerFactory<TextureId>
    + IdentityHandlerFactory<TextureViewId>
    + IdentityHandlerFactory<SamplerId>
    + IdentityHandlerFactory<SurfaceId>
{
}

impl GlobalIdentityHandlerFactory for IdentityManagerFactory {}

pub type Input<G, I> = <<G as IdentityHandlerFactory<I>>::Filter as IdentityHandler<I>>::Input;

pub trait Resource {
    const TYPE: &'static str;
    fn life_guard(&self) -> &crate::LifeGuard;
    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.life_guard().label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

#[derive(Debug)]
pub struct Registry<T: Resource, I: TypedId, F: IdentityHandlerFactory<I>> {
    identity: F::Filter,
    data: RwLock<Storage<T, I>>,
    backend: Backend,
}

impl<T: Resource, I: TypedId, F: IdentityHandlerFactory<I>> Registry<T, I, F> {
    fn new(backend: Backend, factory: &F) -> Self {
        Self {
            identity: factory.spawn(0),
            data: RwLock::new(Storage {
                map: Vec::new(),
                kind: T::TYPE,
                _phantom: PhantomData,
            }),
            backend,
        }
    }

    fn without_backend(factory: &F, kind: &'static str) -> Self {
        Self {
            identity: factory.spawn(1),
            data: RwLock::new(Storage {
                map: Vec::new(),
                kind,
                _phantom: PhantomData,
            }),
            backend: Backend::Empty,
        }
    }
}

#[must_use]
pub(crate) struct FutureId<'a, I: TypedId, T> {
    id: I,
    data: &'a RwLock<Storage<T, I>>,
}

impl<I: TypedId + Copy, T> FutureId<'_, I, T> {
    #[cfg(feature = "trace")]
    pub fn id(&self) -> I {
        self.id
    }

    pub fn into_id(self) -> I {
        self.id
    }

    pub fn assign<'a, A: Access<T>>(self, value: T, _: &'a mut Token<A>) -> Valid<I> {
        self.data.write().insert(self.id, value);
        Valid(self.id)
    }

    pub fn assign_error<'a, A: Access<T>>(self, label: &str, _: &'a mut Token<A>) -> I {
        self.data.write().insert_error(self.id, label);
        self.id
    }
}

impl<T: Resource, I: TypedId + Copy, F: IdentityHandlerFactory<I>> Registry<T, I, F> {
    pub(crate) fn prepare(
        &self,
        id_in: <F::Filter as IdentityHandler<I>>::Input,
    ) -> FutureId<I, T> {
        FutureId {
            id: self.identity.process(id_in, self.backend),
            data: &self.data,
        }
    }

    pub(crate) fn read<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockReadGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.read(), Token::new())
    }

    pub(crate) fn write<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockWriteGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.write(), Token::new())
    }

    pub fn unregister_locked(&self, id: I, guard: &mut Storage<T, I>) -> Option<T> {
        let value = guard.remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        value
    }

    pub(crate) fn unregister<'a, A: Access<T>>(
        &self,
        id: I,
        _token: &'a mut Token<A>,
    ) -> (Option<T>, Token<'a, T>) {
        let value = self.data.write().remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        (value, Token::new())
    }

    pub fn label_for_resource(&self, id: I) -> String {
        let guard = self.data.read();

        let type_name = guard.kind;
        match guard.get(id) {
            Ok(res) => {
                let label = res.label();
                if label.is_empty() {
                    format!("<{}-{:?}>", type_name, id.unzip())
                } else {
                    label.to_string()
                }
            }
            Err(_) => format!(
                "<Invalid-{} label={}>",
                type_name,
                guard.label_for_invalid_id(id)
            ),
        }
    }
}

#[derive(Debug)]
pub struct Hub<B: hal::Backend, F: GlobalIdentityHandlerFactory> {
    pub adapters: Registry<Adapter<B>, AdapterId, F>,
    pub devices: Registry<Device<B>, DeviceId, F>,
    pub swap_chains: Registry<SwapChain<B>, SwapChainId, F>,
    pub pipeline_layouts: Registry<PipelineLayout<B>, PipelineLayoutId, F>,
    pub shader_modules: Registry<ShaderModule<B>, ShaderModuleId, F>,
    pub bind_group_layouts: Registry<BindGroupLayout<B>, BindGroupLayoutId, F>,
    pub bind_groups: Registry<BindGroup<B>, BindGroupId, F>,
    pub command_buffers: Registry<CommandBuffer<B>, CommandBufferId, F>,
    pub render_bundles: Registry<RenderBundle, RenderBundleId, F>,
    pub render_pipelines: Registry<RenderPipeline<B>, RenderPipelineId, F>,
    pub compute_pipelines: Registry<ComputePipeline<B>, ComputePipelineId, F>,
    pub query_sets: Registry<QuerySet<B>, QuerySetId, F>,
    pub buffers: Registry<Buffer<B>, BufferId, F>,
    pub textures: Registry<Texture<B>, TextureId, F>,
    pub texture_views: Registry<TextureView<B>, TextureViewId, F>,
    pub samplers: Registry<Sampler<B>, SamplerId, F>,
}

impl<B: GfxBackend, F: GlobalIdentityHandlerFactory> Hub<B, F> {
    fn new(factory: &F) -> Self {
        Self {
            adapters: Registry::new(B::VARIANT, factory),
            devices: Registry::new(B::VARIANT, factory),
            swap_chains: Registry::new(B::VARIANT, factory),
            pipeline_layouts: Registry::new(B::VARIANT, factory),
            shader_modules: Registry::new(B::VARIANT, factory),
            bind_group_layouts: Registry::new(B::VARIANT, factory),
            bind_groups: Registry::new(B::VARIANT, factory),
            command_buffers: Registry::new(B::VARIANT, factory),
            render_bundles: Registry::new(B::VARIANT, factory),
            render_pipelines: Registry::new(B::VARIANT, factory),
            compute_pipelines: Registry::new(B::VARIANT, factory),
            query_sets: Registry::new(B::VARIANT, factory),
            buffers: Registry::new(B::VARIANT, factory),
            textures: Registry::new(B::VARIANT, factory),
            texture_views: Registry::new(B::VARIANT, factory),
            samplers: Registry::new(B::VARIANT, factory),
        }
    }
}

impl<B: GfxBackend, F: GlobalIdentityHandlerFactory> Hub<B, F> {
    //TODO: instead of having a hacky `with_adapters` parameter,
    // we should have `clear_device(device_id)` that specifically destroys
    // everything related to a logical device.
    fn clear(&self, surface_guard: &mut Storage<Surface, SurfaceId>, with_adapters: bool) {
        use crate::resource::TextureViewInner;
        use hal::{device::Device as _, window::PresentationSurface as _};

        let mut devices = self.devices.data.write();
        for element in devices.map.iter_mut() {
            if let Element::Occupied(ref mut device, _) = *element {
                device.prepare_to_die();
            }
        }

        for element in self.samplers.data.write().map.drain(..) {
            if let Element::Occupied(sampler, _) = element {
                unsafe {
                    devices[sampler.device_id.value]
                        .raw
                        .destroy_sampler(sampler.raw);
                }
            }
        }
        {
            let textures = self.textures.data.read();
            for element in self.texture_views.data.write().map.drain(..) {
                if let Element::Occupied(texture_view, _) = element {
                    match texture_view.inner {
                        TextureViewInner::Native { raw, source_id } => {
                            let device = &devices[textures[source_id.value].device_id.value];
                            unsafe {
                                device.raw.destroy_image_view(raw);
                            }
                        }
                        TextureViewInner::SwapChain { .. } => {} //TODO
                    }
                }
            }
        }

        for element in self.textures.data.write().map.drain(..) {
            if let Element::Occupied(texture, _) = element {
                devices[texture.device_id.value].destroy_texture(texture);
            }
        }
        for element in self.buffers.data.write().map.drain(..) {
            if let Element::Occupied(buffer, _) = element {
                //TODO: unmap if needed
                devices[buffer.device_id.value].destroy_buffer(buffer);
            }
        }
        for element in self.command_buffers.data.write().map.drain(..) {
            if let Element::Occupied(command_buffer, _) = element {
                let device = &devices[command_buffer.device_id.value];
                device
                    .cmd_allocator
                    .after_submit(command_buffer, &device.raw, 0);
            }
        }
        for element in self.bind_groups.data.write().map.drain(..) {
            if let Element::Occupied(bind_group, _) = element {
                let device = &devices[bind_group.device_id.value];
                device.destroy_bind_group(bind_group);
            }
        }

        for element in self.shader_modules.data.write().map.drain(..) {
            if let Element::Occupied(module, _) = element {
                let device = &devices[module.device_id.value];
                unsafe {
                    device.raw.destroy_shader_module(module.raw);
                }
            }
        }
        for element in self.bind_group_layouts.data.write().map.drain(..) {
            if let Element::Occupied(bgl, _) = element {
                let device = &devices[bgl.device_id.value];
                unsafe {
                    device.raw.destroy_descriptor_set_layout(bgl.raw);
                }
            }
        }
        for element in self.pipeline_layouts.data.write().map.drain(..) {
            if let Element::Occupied(pipeline_layout, _) = element {
                let device = &devices[pipeline_layout.device_id.value];
                unsafe {
                    device.raw.destroy_pipeline_layout(pipeline_layout.raw);
                }
            }
        }
        for element in self.compute_pipelines.data.write().map.drain(..) {
            if let Element::Occupied(pipeline, _) = element {
                let device = &devices[pipeline.device_id.value];
                unsafe {
                    device.raw.destroy_compute_pipeline(pipeline.raw);
                }
            }
        }
        for element in self.render_pipelines.data.write().map.drain(..) {
            if let Element::Occupied(pipeline, _) = element {
                let device = &devices[pipeline.device_id.value];
                unsafe {
                    device.raw.destroy_graphics_pipeline(pipeline.raw);
                }
            }
        }

        for (index, element) in self.swap_chains.data.write().map.drain(..).enumerate() {
            if let Element::Occupied(swap_chain, epoch) = element {
                let device = &devices[swap_chain.device_id.value];
                unsafe {
                    device.raw.destroy_semaphore(swap_chain.semaphore);
                }
                let suf_id = TypedId::zip(index as Index, epoch, B::VARIANT);
                //TODO: hold the surface alive by the swapchain
                if surface_guard.contains(suf_id) {
                    let surface = surface_guard.get_mut(suf_id).unwrap();
                    let suf = B::get_surface_mut(surface);
                    unsafe {
                        suf.unconfigure_swapchain(&device.raw);
                    }
                }
            }
        }

        for element in self.query_sets.data.write().map.drain(..) {
            if let Element::Occupied(query_set, _) = element {
                let device = &devices[query_set.device_id.value];
                unsafe {
                    device.raw.destroy_query_pool(query_set.raw);
                }
            }
        }

        for element in devices.map.drain(..) {
            if let Element::Occupied(device, _) = element {
                device.dispose();
            }
        }
        if with_adapters {
            self.adapters.data.write().map.clear();
        }
    }
}

#[derive(Debug)]
pub struct Hubs<F: GlobalIdentityHandlerFactory> {
    #[cfg(vulkan)]
    vulkan: Hub<backend::Vulkan, F>,
    #[cfg(metal)]
    metal: Hub<backend::Metal, F>,
    #[cfg(dx12)]
    dx12: Hub<backend::Dx12, F>,
    #[cfg(dx11)]
    dx11: Hub<backend::Dx11, F>,
    #[cfg(gl)]
    gl: Hub<backend::Gl, F>,
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
pub struct Global<G: GlobalIdentityHandlerFactory> {
    pub instance: Instance,
    pub surfaces: Registry<Surface, SurfaceId, G>,
    hubs: Hubs<G>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn new(name: &str, factory: G, backends: wgt::BackendBit) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, 1, backends),
            surfaces: Registry::without_backend(&factory, "Surface"),
            hubs: Hubs::new(&factory),
        }
    }

    pub fn clear_backend<B: GfxBackend>(&self, _dummy: ()) {
        let mut surface_guard = self.surfaces.data.write();
        let hub = B::hub(self);
        // this is used for tests, which keep the adapter
        hub.clear(&mut *surface_guard, false);
    }
}

impl<G: GlobalIdentityHandlerFactory> Drop for Global<G> {
    fn drop(&mut self) {
        if !thread::panicking() {
            log::info!("Dropping Global");
            let mut surface_guard = self.surfaces.data.write();

            // destroy hubs
            #[cfg(vulkan)]
            {
                self.hubs.vulkan.clear(&mut *surface_guard, true);
            }
            #[cfg(metal)]
            {
                self.hubs.metal.clear(&mut *surface_guard, true);
            }
            #[cfg(dx12)]
            {
                self.hubs.dx12.clear(&mut *surface_guard, true);
            }
            #[cfg(dx11)]
            {
                self.hubs.dx11.clear(&mut *surface_guard, true);
            }
            #[cfg(gl)]
            {
                self.hubs.gl.clear(&mut *surface_guard, true);
            }

            // destroy surfaces
            for element in surface_guard.map.drain(..) {
                if let Element::Occupied(surface, _) = element {
                    self.instance.destroy_surface(surface);
                }
            }
        }
    }
}

pub trait GfxBackend: hal::Backend {
    const VARIANT: Backend;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G>;
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface;
}

#[cfg(vulkan)]
impl GfxBackend for backend::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.vulkan
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.vulkan.as_mut().unwrap()
    }
}

#[cfg(metal)]
impl GfxBackend for backend::Metal {
    const VARIANT: Backend = Backend::Metal;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.metal
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.metal.as_mut().unwrap()
    }
}

#[cfg(dx12)]
impl GfxBackend for backend::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.dx12
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.dx12.as_mut().unwrap()
    }
}

#[cfg(dx11)]
impl GfxBackend for backend::Dx11 {
    const VARIANT: Backend = Backend::Dx11;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.dx11
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.dx11.as_mut().unwrap()
    }
}

#[cfg(gl)]
impl GfxBackend for backend::Gl {
    const VARIANT: Backend = Backend::Gl;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self, G> {
        &global.hubs.gl
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.gl.as_mut().unwrap()
    }
}

#[cfg(test)]
fn _test_send_sync(global: &Global<IdentityManagerFactory>) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
