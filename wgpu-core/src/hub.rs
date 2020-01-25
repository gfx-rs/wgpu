/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend,
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::CommandBuffer,
    device::{Device, ShaderModule},
    id::{
        AdapterId,
        BindGroupId,
        BindGroupLayoutId,
        BufferId,
        CommandBufferId,
        ComputePipelineId,
        DeviceId,
        PipelineLayoutId,
        RenderPipelineId,
        SamplerId,
        ShaderModuleId,
        SurfaceId,
        SwapChainId,
        TextureId,
        TextureViewId,
        TypedId,
    },
    instance::{Adapter, Instance, Surface},
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{Buffer, Sampler, Texture, TextureView},
    swap_chain::SwapChain,
    Backend,
    Epoch,
    Index,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use vec_map::VecMap;

#[cfg(debug_assertions)]
use std::cell::Cell;
use std::{fmt::Debug, marker::PhantomData, ops};


/// A simple structure to manage identities of objects.
#[derive(Debug)]
pub struct IdentityManager {
    free: Vec<Index>,
    epochs: Vec<Epoch>,
}

impl Default for IdentityManager {
    fn default() -> Self {
        IdentityManager {
            free: Default::default(),
            epochs: Default::default(),
        }
    }
}

impl IdentityManager {
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

    pub fn free<I: TypedId>(&mut self, id: I) {
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
pub struct Storage<T, I: TypedId> {
    //TODO: consider concurrent hashmap?
    map: VecMap<(T, Epoch)>,
    _phantom: PhantomData<I>,
}

impl<T, I: TypedId> ops::Index<I> for Storage<T, I> {
    type Output = T;
    fn index(&self, id: I) -> &T {
        let (index, epoch, _) = id.unzip();
        let (ref value, storage_epoch) = self.map[index as usize];
        assert_eq!(epoch, storage_epoch);
        value
    }
}

impl<T, I: TypedId> ops::IndexMut<I> for Storage<T, I> {
    fn index_mut(&mut self, id: I) -> &mut T {
        let (index, epoch, _) = id.unzip();
        let (ref mut value, storage_epoch) = self.map[index as usize];
        assert_eq!(epoch, storage_epoch);
        value
    }
}

impl<T, I: TypedId> Storage<T, I> {
    pub fn contains(&self, id: I) -> bool {
        let (index, epoch, _) = id.unzip();
        match self.map.get(index as usize) {
            Some(&(_, storage_epoch)) => epoch == storage_epoch,
            None => false,
        }
    }

    pub fn insert(&mut self, id: I, value: T) -> Option<T> {
        let (index, epoch, _) = id.unzip();
        let old = self.map.insert(index as usize, (value, epoch));
        old.map(|(v, _storage_epoch)| v)
    }

    pub fn remove(&mut self, id: I) -> Option<T> {
        let (index, epoch, _) = id.unzip();
        self.map
            .remove(index as usize)
            .map(|(value, storage_epoch)| {
                assert_eq!(epoch, storage_epoch);
                value
            })
    }

    pub fn iter(&self, backend: Backend) -> impl Iterator<Item = (I, &T)> {
        self.map
            .iter()
            .map(move |(index, (value, storage_epoch))| {
                (I::zip(index as Index, *storage_epoch, backend), value)
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
impl<B: hal::Backend> Access<PipelineLayout<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Root {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for Root {}
impl<B: hal::Backend> Access<BindGroup<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Root {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for SwapChain<B> {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for Root {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for Root {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<ShaderModule<B>> for Root {}
impl<B: hal::Backend> Access<ShaderModule<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for Root {}
impl<B: hal::Backend> Access<Buffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for ComputePipeline<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for RenderPipeline<B> {}
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
pub struct Token<'a, T: 'a> {
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
        Token { level: PhantomData }
    }
}

impl Token<'static, Root> {
    pub fn root() -> Self {
        #[cfg(debug_assertions)]
        ACTIVE_TOKEN.with(|active| {
            assert_eq!(0, active.replace(1), "Root token is already active");
        });

        Token { level: PhantomData }
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


pub trait IdentityFilter<I>: Debug {
    type Input: Clone + Debug;
    fn process(&self, id: Self::Input, backend: Backend) -> I;
    fn free(&self, id: I);
}

impl<I: TypedId + Clone + Debug> IdentityFilter<I> for () {
    type Input = I;
    fn process(&self, id: I, _backend: Backend) -> I {
        //debug_assert_eq!(id.unzip().2, backend);
        id
    }
    fn free(&self, _id: I) {}
}

impl<I: TypedId + Debug> IdentityFilter<I> for Mutex<IdentityManager> {
    type Input = PhantomData<I>;
    fn process(&self, _id: Self::Input, backend: Backend) -> I {
        self.lock().alloc(backend)
    }
    fn free(&self, id: I) {
        self.lock().free(id)
    }
}

/// Compound trait for all the things a device cares about
/// for the matter of destruction/cleanup.
pub trait AllIdentityFilter:
    IdentityFilter<BufferId>
    + IdentityFilter<TextureId>
    + IdentityFilter<TextureViewId>
    + IdentityFilter<BindGroupId>
    + IdentityFilter<SamplerId>
{
}

impl AllIdentityFilter for Mutex<IdentityManager> {}
impl AllIdentityFilter for () {}


#[derive(Debug)]
pub struct Registry<T, I: TypedId, F> {
    pub(crate) identity: F,
    data: RwLock<Storage<T, I>>,
    backend: Backend,
}

impl<T, I: TypedId, F: Default> Registry<T, I, F> {
    fn new(backend: Backend) -> Self {
        Registry {
            identity: F::default(),
            data: RwLock::new(Storage {
                map: VecMap::new(),
                _phantom: PhantomData,
            }),
            backend,
        }
    }
}

impl<T, I: TypedId + Copy, F> Registry<T, I, F> {
    pub fn register<A: Access<T>>(&self, id: I, value: T, _token: &mut Token<A>) {
        debug_assert_eq!(id.unzip().2, self.backend);
        let old = self.data.write().insert(id, value);
        assert!(old.is_none());
    }

    pub fn read<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockReadGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.read(), Token::new())
    }

    pub fn write<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockWriteGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.write(), Token::new())
    }
}

impl<T, I: TypedId + Copy, F: IdentityFilter<I>> Registry<T, I, F> {
    pub fn register_identity<A: Access<T>>(
        &self,
        id_in: F::Input,
        value: T,
        token: &mut Token<A>,
    ) -> I {
        let id = self.identity.process(id_in, self.backend);
        self.register(id, value, token);
        id
    }

    pub fn unregister<'a, A: Access<T>>(
        &self,
        id: I,
        _token: &'a mut Token<A>,
    ) -> (T, Token<'a, T>) {
        let value = self.data.write().remove(id).unwrap();
        //Note: careful about the order here!
        self.identity.free(id);
        (value, Token::new())
    }
}

#[derive(Debug)]
pub struct Hub<B: hal::Backend, F> {
    pub adapters: Registry<Adapter<B>, AdapterId, F>,
    pub devices: Registry<Device<B>, DeviceId, F>,
    pub swap_chains: Registry<SwapChain<B>, SwapChainId, F>,
    pub pipeline_layouts: Registry<PipelineLayout<B>, PipelineLayoutId, F>,
    pub shader_modules: Registry<ShaderModule<B>, ShaderModuleId, F>,
    pub bind_group_layouts: Registry<BindGroupLayout<B>, BindGroupLayoutId, F>,
    pub bind_groups: Registry<BindGroup<B>, BindGroupId, F>,
    pub command_buffers: Registry<CommandBuffer<B>, CommandBufferId, F>,
    pub render_pipelines: Registry<RenderPipeline<B>, RenderPipelineId, F>,
    pub compute_pipelines: Registry<ComputePipeline<B>, ComputePipelineId, F>,
    pub buffers: Registry<Buffer<B>, BufferId, F>,
    pub textures: Registry<Texture<B>, TextureId, F>,
    pub texture_views: Registry<TextureView<B>, TextureViewId, F>,
    pub samplers: Registry<Sampler<B>, SamplerId, F>,
}

impl<B: GfxBackend, F: Default> Default for Hub<B, F> {
    fn default() -> Self {
        Hub {
            adapters: Registry::new(B::VARIANT),
            devices: Registry::new(B::VARIANT),
            swap_chains: Registry::new(B::VARIANT),
            pipeline_layouts: Registry::new(B::VARIANT),
            shader_modules: Registry::new(B::VARIANT),
            bind_group_layouts: Registry::new(B::VARIANT),
            bind_groups: Registry::new(B::VARIANT),
            command_buffers: Registry::new(B::VARIANT),
            render_pipelines: Registry::new(B::VARIANT),
            compute_pipelines: Registry::new(B::VARIANT),
            buffers: Registry::new(B::VARIANT),
            textures: Registry::new(B::VARIANT),
            texture_views: Registry::new(B::VARIANT),
            samplers: Registry::new(B::VARIANT),
        }
    }
}

impl<B: hal::Backend, F> Drop for Hub<B, F> {
    fn drop(&mut self) {
        use crate::resource::TextureViewInner;
        use hal::device::Device as _;

        let mut devices = self.devices.data.write();

        for (_, (sampler, _)) in self.samplers.data.write().map.drain() {
            unsafe {
                devices[sampler.device_id.value]
                    .raw
                    .destroy_sampler(sampler.raw);
            }
        }
        {
            let textures = self.textures.data.read();
            for (_, (texture_view, _)) in self.texture_views.data.write().map.drain() {
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
        for (_, (texture, _)) in self.textures.data.write().map.drain() {
            devices[texture.device_id.value].destroy_texture(texture);
        }
        for (_, (buffer, _)) in self.buffers.data.write().map.drain() {
            //TODO: unmap if needed
            devices[buffer.device_id.value].destroy_buffer(buffer);
        }
        for (_, (command_buffer, _)) in self.command_buffers.data.write().map.drain() {
            devices[command_buffer.device_id.value]
                .com_allocator
                .after_submit(command_buffer, 0);
        }
        for (_, (bind_group, _)) in self.bind_groups.data.write().map.drain() {
            let device = &devices[bind_group.device_id.value];
            device.destroy_bind_group(bind_group);
        }

        //TODO:
        // self.compute_pipelines
        // self.render_pipelines
        // self.bind_group_layouts
        // self.pipeline_layouts
        // self.shader_modules
        // self.swap_chains
        // self.adapters

        for (_, (device, _)) in devices.map.drain() {
            device.dispose();
        }
    }
}

#[derive(Debug, Default)]
pub struct Hubs<F> {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    vulkan: Hub<backend::Vulkan, F>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    metal: Hub<backend::Metal, F>,
    #[cfg(windows)]
    dx12: Hub<backend::Dx12, F>,
    #[cfg(windows)]
    dx11: Hub<backend::Dx11, F>,
}

#[derive(Debug)]
pub struct Global<F> {
    pub instance: Instance,
    pub surfaces: Registry<Surface, SurfaceId, F>,
    hubs: Hubs<F>,
}

impl<F: Default> Global<F> {
    pub fn new(name: &str) -> Self {
        Global {
            instance: Instance::new(name, 1),
            surfaces: Registry::new(Backend::Empty),
            hubs: Hubs::default(),
        }
    }

    pub fn delete(self) {
        let Global {
            mut instance,
            surfaces,
            hubs,
        } = self;
        drop(hubs);
        // destroy surfaces
        for (_, (surface, _)) in surfaces.data.write().map.drain() {
            instance.destroy_surface(surface);
        }
    }
}

pub trait GfxBackend: hal::Backend {
    const VARIANT: Backend;
    fn hub<F>(global: &Global<F>) -> &Hub<Self, F>;
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface;
}

#[cfg(any(
    not(any(target_os = "ios", target_os = "macos")),
    feature = "gfx-backend-vulkan"
))]
impl GfxBackend for backend::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
    fn hub<F>(global: &Global<F>) -> &Hub<Self, F> {
        &global.hubs.vulkan
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.vulkan.as_mut().unwrap()
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
impl GfxBackend for backend::Metal {
    const VARIANT: Backend = Backend::Metal;
    fn hub<F>(global: &Global<F>) -> &Hub<Self, F> {
        &global.hubs.metal
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        &mut surface.metal
    }
}

#[cfg(windows)]
impl GfxBackend for backend::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
    fn hub<F>(global: &Global<F>) -> &Hub<Self, F> {
        &global.hubs.dx12
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.dx12.as_mut().unwrap()
    }
}

#[cfg(windows)]
impl GfxBackend for backend::Dx11 {
    const VARIANT: Backend = Backend::Dx11;
    fn hub<F>(global: &Global<F>) -> &Hub<Self, F> {
        &global.hubs.dx11
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        &mut surface.dx11
    }
}
