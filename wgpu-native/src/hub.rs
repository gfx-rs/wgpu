use crate::{
    AdapterHandle,
    AdapterId,
    BindGroupHandle,
    BindGroupId,
    BindGroupLayoutHandle,
    BindGroupLayoutId,
    BufferHandle,
    BufferId,
    CommandBufferHandle,
    CommandBufferId,
    ComputePassHandle,
    ComputePassId,
    ComputePipelineHandle,
    ComputePipelineId,
    DeviceHandle,
    DeviceId,
    Epoch,
    Index,
    PipelineLayoutHandle,
    PipelineLayoutId,
    RenderPassHandle,
    RenderPassId,
    RenderPipelineHandle,
    RenderPipelineId,
    SamplerHandle,
    SamplerId,
    ShaderModuleHandle,
    ShaderModuleId,
    SurfaceHandle,
    SurfaceId,
    TextureHandle,
    TextureId,
    TextureViewHandle,
    TextureViewId,
    TypedId,
};
#[cfg(not(feature = "gfx-backend-gl"))]
use crate::{InstanceHandle, InstanceId};

use lazy_static::lazy_static;
#[cfg(feature = "local")]
use parking_lot::Mutex;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use vec_map::VecMap;

#[allow(unused)]
use std::cell::Cell;
use std::{
    marker::PhantomData,
    ops,
    sync::Arc,
};


/// A simple structure to manage identities of objects.
#[derive(Debug)]
pub struct IdentityManager<I: TypedId> {
    free: Vec<Index>,
    epochs: Vec<Epoch>,
    phantom: PhantomData<I>,
}

impl<I: TypedId> Default for IdentityManager<I> {
    fn default() -> IdentityManager<I> {
        IdentityManager {
            free: Default::default(),
            epochs: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<I: TypedId> IdentityManager<I> {
    pub fn alloc(&mut self) -> I {
        match self.free.pop() {
            Some(index) => I::new(index, self.epochs[index as usize]),
            None => {
                let id = I::new(self.epochs.len() as Index, 1);
                self.epochs.push(id.epoch());
                id
            }
        }
    }

    pub fn free(&mut self, id: I) {
        let (index, epoch) = (id.index(), id.epoch());
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
        let (ref value, epoch) = self.map[id.index() as usize];
        assert_eq!(epoch, id.epoch());
        value
    }
}

impl<T, I: TypedId> ops::IndexMut<I> for Storage<T, I> {
    fn index_mut(&mut self, id: I) -> &mut T {
        let (ref mut value, epoch) = self.map[id.index() as usize];
        assert_eq!(epoch, id.epoch());
        value
    }
}

impl<T, I: TypedId> Storage<T, I> {
    pub fn contains(&self, id: I) -> bool {
        match self.map.get(id.index() as usize) {
            Some(&(_, epoch)) if epoch == id.epoch() => true,
            _ => false,
        }
    }

    pub fn remove(&mut self, id: I) -> T {
        let (value, epoch) = self.map.remove(id.index() as usize).unwrap();
        assert_eq!(epoch, id.epoch());
        value
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
#[cfg(not(feature = "gfx-backend-gl"))]
impl Access<InstanceHandle> for Root {}
impl Access<SurfaceHandle> for Root {}
#[cfg(not(feature = "gfx-backend-gl"))]
impl Access<SurfaceHandle> for InstanceHandle {}
impl Access<AdapterHandle> for Root {}
impl Access<AdapterHandle> for SurfaceHandle {}
impl Access<DeviceHandle> for Root {}
impl Access<DeviceHandle> for SurfaceHandle {}
impl Access<DeviceHandle> for AdapterHandle {}
impl Access<PipelineLayoutHandle> for Root {}
impl Access<PipelineLayoutHandle> for DeviceHandle {}
impl Access<BindGroupLayoutHandle> for Root {}
impl Access<BindGroupLayoutHandle> for DeviceHandle {}
impl Access<BindGroupHandle> for Root {}
impl Access<BindGroupHandle> for DeviceHandle {}
impl Access<BindGroupHandle> for PipelineLayoutHandle {}
impl Access<BindGroupHandle> for CommandBufferHandle {}
impl Access<CommandBufferHandle> for Root {}
impl Access<CommandBufferHandle> for DeviceHandle {}
impl Access<ComputePassHandle> for Root {}
impl Access<ComputePassHandle> for BindGroupHandle {}
impl Access<ComputePassHandle> for CommandBufferHandle {}
impl Access<RenderPassHandle> for Root {}
impl Access<RenderPassHandle> for BindGroupHandle {}
impl Access<RenderPassHandle> for CommandBufferHandle {}
impl Access<ComputePipelineHandle> for Root {}
impl Access<ComputePipelineHandle> for ComputePassHandle {}
impl Access<RenderPipelineHandle> for Root {}
impl Access<RenderPipelineHandle> for RenderPassHandle {}
impl Access<ShaderModuleHandle> for Root {}
impl Access<ShaderModuleHandle> for PipelineLayoutHandle {}
impl Access<BufferHandle> for Root {}
impl Access<BufferHandle> for DeviceHandle {}
impl Access<BufferHandle> for BindGroupLayoutHandle {}
impl Access<BufferHandle> for BindGroupHandle {}
impl Access<BufferHandle> for CommandBufferHandle {}
impl Access<BufferHandle> for ComputePassHandle {}
impl Access<BufferHandle> for ComputePipelineHandle {}
impl Access<BufferHandle> for RenderPassHandle {}
impl Access<BufferHandle> for RenderPipelineHandle {}
impl Access<TextureHandle> for Root {}
impl Access<TextureHandle> for DeviceHandle {}
impl Access<TextureHandle> for BufferHandle {}
impl Access<TextureViewHandle> for Root {}
impl Access<TextureViewHandle> for DeviceHandle {}
impl Access<TextureViewHandle> for TextureHandle {}
impl Access<SamplerHandle> for Root {}
impl Access<SamplerHandle> for TextureViewHandle {}

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
        Token {
            level: PhantomData,
        }
    }
}

impl Token<'static, Root> {
    pub fn root() -> Self {
        #[cfg(debug_assertions)]
        ACTIVE_TOKEN.with(|active| {
            assert_eq!(0, active.replace(1), "Root token is already active");
        });

        Token {
            level: PhantomData,
        }
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
}

impl<T, I: TypedId> Default for Registry<T, I> {
    fn default() -> Self {
        Registry {
            #[cfg(feature = "local")]
            identity: Mutex::new(IdentityManager::default()),
            data: RwLock::new(Storage {
                map: VecMap::new(),
                _phantom: PhantomData,
            }),
        }
    }
}

impl<T, I: TypedId + Copy> Registry<T, I> {
    pub fn register<A: Access<T>>(
        &self, id: I, value: T, _token: &mut Token<A>
    ) {
        let old = self
            .data
            .write()
            .map
            .insert(id.index() as usize, (value, id.epoch()));
        assert!(old.is_none());
    }

    #[cfg(feature = "local")]
    pub fn register_local<A: Access<T>>(
        &self, value: T, token: &mut Token<A>
    ) -> I {
        let id = self.identity.lock().alloc();
        self.register(id, value, token);
        id
    }

    pub fn unregister<A: Access<T>>(
        &self, id: I, _token: &mut Token<A>
    ) -> (T, Token<T>) {
        let value = self.data.write().remove(id);
        //Note: careful about the order here!
        #[cfg(feature = "local")]
        self.identity.lock().free(id);
        (value, Token::new())
    }

    pub fn read<A: Access<T>>(
        &self, _token: &mut Token<A>
    ) -> (RwLockReadGuard<Storage<T, I>>, Token<T>) {
        (self.data.read(), Token::new())
    }

    pub fn write<A: Access<T>>(
        &self, _token: &mut Token<A>
    ) -> (RwLockWriteGuard<Storage<T, I>>, Token<T>) {
        (self.data.write(), Token::new())
    }
}

#[derive(Default, Debug)]
pub struct Hub {
    #[cfg(not(feature = "gfx-backend-gl"))]
    pub instances: Arc<Registry<InstanceHandle, InstanceId>>,
    pub surfaces: Arc<Registry<SurfaceHandle, SurfaceId>>,
    pub adapters: Arc<Registry<AdapterHandle, AdapterId>>,
    pub devices: Arc<Registry<DeviceHandle, DeviceId>>,
    pub pipeline_layouts: Arc<Registry<PipelineLayoutHandle, PipelineLayoutId>>,
    pub shader_modules: Arc<Registry<ShaderModuleHandle, ShaderModuleId>>,
    pub bind_group_layouts: Arc<Registry<BindGroupLayoutHandle, BindGroupLayoutId>>,
    pub bind_groups: Arc<Registry<BindGroupHandle, BindGroupId>>,
    pub command_buffers: Arc<Registry<CommandBufferHandle, CommandBufferId>>,
    pub render_passes: Arc<Registry<RenderPassHandle, RenderPassId>>,
    pub render_pipelines: Arc<Registry<RenderPipelineHandle, RenderPipelineId>>,
    pub compute_passes: Arc<Registry<ComputePassHandle, ComputePassId>>,
    pub compute_pipelines: Arc<Registry<ComputePipelineHandle, ComputePipelineId>>,
    pub buffers: Arc<Registry<BufferHandle, BufferId>>,
    pub textures: Arc<Registry<TextureHandle, TextureId>>,
    pub texture_views: Arc<Registry<TextureViewHandle, TextureViewId>>,
    pub samplers: Arc<Registry<SamplerHandle, SamplerId>>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
