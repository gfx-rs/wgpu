/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend,
    id::{Input, Output},
    Adapter,
    AdapterId,
    Backend,
    BindGroup,
    BindGroupId,
    BindGroupLayout,
    BindGroupLayoutId,
    Buffer,
    BufferId,
    CommandBuffer,
    CommandBufferId,
    ComputePass,
    ComputePassId,
    ComputePipeline,
    ComputePipelineId,
    Device,
    DeviceId,
    Epoch,
    Index,
    Instance,
    PipelineLayout,
    PipelineLayoutId,
    RenderPass,
    RenderPassId,
    RenderPipeline,
    RenderPipelineId,
    Sampler,
    SamplerId,
    ShaderModule,
    ShaderModuleId,
    Surface,
    SurfaceId,
    SwapChain,
    SwapChainId,
    Texture,
    TextureId,
    TextureView,
    TextureViewId,
    TypedId,
};

#[cfg(feature = "local")]
use parking_lot::Mutex;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use vec_map::VecMap;

#[cfg(debug_assertions)]
use std::cell::Cell;
#[cfg(feature = "local")]
use std::sync::Arc;
use std::{marker::PhantomData, ops};


/// A simple structure to manage identities of objects.
#[derive(Debug)]
pub struct IdentityManager<I: TypedId> {
    free: Vec<Index>,
    epochs: Vec<Epoch>,
    backend: Backend,
    phantom: PhantomData<I>,
}

impl<I: TypedId> IdentityManager<I> {
    pub fn new(backend: Backend) -> Self {
        IdentityManager {
            free: Default::default(),
            epochs: Default::default(),
            backend,
            phantom: PhantomData,
        }
    }
}

impl<I: TypedId> IdentityManager<I> {
    pub fn alloc(&mut self) -> I {
        match self.free.pop() {
            Some(index) => I::zip(index, self.epochs[index as usize], self.backend),
            None => {
                let epoch = 1;
                let id = I::zip(self.epochs.len() as Index, epoch, self.backend);
                self.epochs.push(epoch);
                id
            }
        }
    }

    pub fn free(&mut self, id: I) {
        let (index, epoch, backend) = id.unzip();
        debug_assert_eq!(backend, self.backend);
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
impl<B: hal::Backend> Access<SwapChain<B>> for Device<B> {}
impl<B: hal::Backend> Access<PipelineLayout<B>> for Root {}
impl<B: hal::Backend> Access<PipelineLayout<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Root {}
impl<B: hal::Backend> Access<BindGroupLayout<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for Root {}
impl<B: hal::Backend> Access<BindGroup<B>> for Device<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<BindGroup<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Root {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<CommandBuffer<B>> for SwapChain<B> {}
impl<B: hal::Backend> Access<ComputePass<B>> for Root {}
impl<B: hal::Backend> Access<ComputePass<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<ComputePass<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<RenderPass<B>> for Root {}
impl<B: hal::Backend> Access<RenderPass<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<RenderPass<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for Root {}
impl<B: hal::Backend> Access<ComputePipeline<B>> for ComputePass<B> {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for Root {}
impl<B: hal::Backend> Access<RenderPipeline<B>> for RenderPass<B> {}
impl<B: hal::Backend> Access<ShaderModule<B>> for Root {}
impl<B: hal::Backend> Access<ShaderModule<B>> for PipelineLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for Root {}
impl<B: hal::Backend> Access<Buffer<B>> for Device<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroupLayout<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for BindGroup<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for CommandBuffer<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for ComputePass<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for ComputePipeline<B> {}
impl<B: hal::Backend> Access<Buffer<B>> for RenderPass<B> {}
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


#[derive(Debug)]
pub struct Registry<T, I: TypedId> {
    #[cfg(feature = "local")]
    pub identity: Mutex<IdentityManager<I>>,
    data: RwLock<Storage<T, I>>,
    backend: Backend,
}

impl<T, I: TypedId> Registry<T, I> {
    fn new(backend: Backend) -> Self {
        Registry {
            #[cfg(feature = "local")]
            identity: Mutex::new(IdentityManager::new(backend)),
            data: RwLock::new(Storage {
                map: VecMap::new(),
                _phantom: PhantomData,
            }),
            backend,
        }
    }
}

impl<T, I: TypedId + Copy> Registry<T, I> {
    pub fn register<A: Access<T>>(&self, id: I, value: T, _token: &mut Token<A>) {
        debug_assert_eq!(id.unzip().2, self.backend);
        let old = self.data.write().insert(id, value);
        assert!(old.is_none());
    }

    #[cfg(feature = "local")]
    pub fn new_identity(&self, _id_in: Input<I>) -> (I, Output<I>) {
        let id = self.identity.lock().alloc();
        (id, id)
    }

    #[cfg(not(feature = "local"))]
    pub fn new_identity(&self, id_in: Input<I>) -> (I, Output<I>) {
        //TODO: debug_assert_eq!(self.backend, id_in.backend());
        (id_in, PhantomData)
    }

    pub fn register_identity<A: Access<T>>(
        &self,
        id_in: Input<I>,
        value: T,
        token: &mut Token<A>,
    ) -> Output<I> {
        let (id, output) = self.new_identity(id_in);
        self.register(id, value, token);
        output
    }

    pub fn unregister<A: Access<T>>(&self, id: I, _token: &mut Token<A>) -> (T, Token<T>) {
        let value = self.data.write().remove(id).unwrap();
        //Note: careful about the order here!
        #[cfg(feature = "local")]
        self.identity.lock().free(id);
        (value, Token::new())
    }

    pub fn read<A: Access<T>>(
        &self,
        _token: &mut Token<A>,
    ) -> (RwLockReadGuard<Storage<T, I>>, Token<T>) {
        (self.data.read(), Token::new())
    }

    pub fn write<A: Access<T>>(
        &self,
        _token: &mut Token<A>,
    ) -> (RwLockWriteGuard<Storage<T, I>>, Token<T>) {
        (self.data.write(), Token::new())
    }
}

#[derive(Debug)]
pub struct Hub<B: hal::Backend> {
    pub adapters: Registry<Adapter<B>, AdapterId>,
    pub devices: Registry<Device<B>, DeviceId>,
    pub swap_chains: Registry<SwapChain<B>, SwapChainId>,
    pub pipeline_layouts: Registry<PipelineLayout<B>, PipelineLayoutId>,
    pub shader_modules: Registry<ShaderModule<B>, ShaderModuleId>,
    pub bind_group_layouts: Registry<BindGroupLayout<B>, BindGroupLayoutId>,
    pub bind_groups: Registry<BindGroup<B>, BindGroupId>,
    pub command_buffers: Registry<CommandBuffer<B>, CommandBufferId>,
    pub render_passes: Registry<RenderPass<B>, RenderPassId>,
    pub render_pipelines: Registry<RenderPipeline<B>, RenderPipelineId>,
    pub compute_passes: Registry<ComputePass<B>, ComputePassId>,
    pub compute_pipelines: Registry<ComputePipeline<B>, ComputePipelineId>,
    pub buffers: Registry<Buffer<B>, BufferId>,
    pub textures: Registry<Texture<B>, TextureId>,
    pub texture_views: Registry<TextureView<B>, TextureViewId>,
    pub samplers: Registry<Sampler<B>, SamplerId>,
}

impl<B: GfxBackend> Default for Hub<B> {
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
            render_passes: Registry::new(B::VARIANT),
            render_pipelines: Registry::new(B::VARIANT),
            compute_passes: Registry::new(B::VARIANT),
            compute_pipelines: Registry::new(B::VARIANT),
            buffers: Registry::new(B::VARIANT),
            textures: Registry::new(B::VARIANT),
            texture_views: Registry::new(B::VARIANT),
            samplers: Registry::new(B::VARIANT),
        }
    }
}

#[derive(Debug, Default)]
pub struct Hubs {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    vulkan: Hub<backend::Vulkan>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    metal: Hub<backend::Metal>,
    #[cfg(windows)]
    dx12: Hub<backend::Dx12>,
    #[cfg(windows)]
    dx11: Hub<backend::Dx11>,
}

#[derive(Debug)]
pub struct Global {
    pub instance: Instance,
    pub surfaces: Registry<Surface, SurfaceId>,
    hubs: Hubs,
}

#[cfg(feature = "local")]
lazy_static::lazy_static! {
    pub static ref GLOBAL: Arc<Global> = Arc::new(Global {
        instance: Instance::new("wgpu", 1),
        surfaces: Registry::new(Backend::Empty),
        hubs: Hubs::default(),
    });
}

pub trait GfxBackend: hal::Backend {
    const VARIANT: Backend;
    fn hub(global: &Global) -> &Hub<Self>;
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface;
}

#[cfg(any(
    not(any(target_os = "ios", target_os = "macos")),
    feature = "gfx-backend-vulkan"
))]
impl GfxBackend for backend::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.vulkan
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.vulkan.as_mut().unwrap()
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
impl GfxBackend for backend::Metal {
    const VARIANT: Backend = Backend::Metal;
    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.metal
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        &mut surface.metal
    }
}

#[cfg(windows)]
impl GfxBackend for backend::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.dx12
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        surface.dx12.as_mut().unwrap()
    }
}

#[cfg(windows)]
impl GfxBackend for backend::Dx11 {
    const VARIANT: Backend = Backend::Dx11;
    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.dx11
    }
    fn get_surface_mut(surface: &mut Surface) -> &mut Self::Surface {
        &mut surface.dx11
    }
}
