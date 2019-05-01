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
    InstanceHandle,
    InstanceId,
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
    TypedId
};

use lazy_static::lazy_static;
#[cfg(feature = "local")]
use parking_lot::Mutex;
use parking_lot::RwLock;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use vec_map::VecMap;

use std::{ops, sync::Arc};

pub(crate) type Index = u32;
pub(crate) type Epoch = u32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Id(Index, Epoch);

pub trait NewId {
    fn new(index: Index, epoch: Epoch) -> Self;
    fn index(&self) -> Index;
    fn epoch(&self) -> Epoch;
}

impl NewId for Id {
    fn new(index: Index, epoch: Epoch) -> Self {
        Id(index, epoch)
    }

    fn index(&self) -> Index {
        self.0
    }

    fn epoch(&self) -> Epoch {
        self.1
    }
}

/// A simple structure to manage identities of objects.
#[derive(Default)]
pub struct IdentityManager {
    free: Vec<Index>,
    epochs: Vec<Epoch>,
}

impl IdentityManager {
    pub fn alloc(&mut self) -> Id {
        match self.free.pop() {
            Some(index) => Id(index, self.epochs[index as usize]),
            None => {
                let id = Id(self.epochs.len() as Index, 1);
                self.epochs.push(id.1);
                id
            }
        }
    }

    pub fn free(&mut self, Id(index, epoch): Id) {
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

pub struct Storage<T, I:TypedId> {
    //TODO: consider concurrent hashmap?
    map: VecMap<(T, Epoch)>,
    _phantom: std::marker::PhantomData<I>,
}

impl<T, I:TypedId> ops::Index<I> for Storage<T, I> {
    type Output = T;
    fn index(&self, id: I) -> &T {
        let (ref value, epoch) = self.map[id.raw().index() as usize];
        assert_eq!(epoch, id.raw().1);
        value
    }
}

impl<T, I:TypedId> ops::IndexMut<I> for Storage<T, I> {
    fn index_mut(&mut self, id: I) -> &mut T {
        let (ref mut value, epoch) = self.map[id.raw().index() as usize];
        assert_eq!(epoch, id.raw().1);
        value
    }
}

impl<T, I:TypedId> Storage<T, I> {
    pub fn contains(&self, id: I) -> bool {
        match self.map.get(id.raw().index() as usize) {
            Some(&(_, epoch)) if epoch == id.raw().1 => true,
            _ => false,
        }
    }
}

pub struct Registry<T, I:TypedId> {
    #[cfg(feature = "local")]
    identity: Mutex<IdentityManager>,
    data: RwLock<Storage<T, I>>,
}

impl<T, I: TypedId> Default for Registry<T, I> {
    fn default() -> Self {
        Registry {
            #[cfg(feature = "local")]
            identity: Mutex::new(IdentityManager::default()),
            data: RwLock::new(Storage { map: VecMap::new(), _phantom: std::marker::PhantomData }),
        }
    }
}

impl<T, I: TypedId> ops::Deref for Registry<T, I> {
    type Target = RwLock<Storage<T, I>>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, I: TypedId> ops::DerefMut for Registry<T, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, I: TypedId + Clone> Registry<T, I> {
    pub fn register(&self, id: I, value: T) {
        let old = self.data.write().map.insert(id.raw().0 as usize, (value, id.raw().1));
        assert!(old.is_none());
    }

    #[cfg(feature = "local")]
    pub fn register_local(&self, value: T) -> I {
        let raw_id = self.identity.lock().alloc();
        let id:I = raw_id.into();
        self.register(id.clone(), value);
        id
    }

    pub fn unregister(&self, id: I) -> T {
        #[cfg(feature = "local")]
        self.identity.lock().free(id.raw());
        let (value, epoch) = self.data.write().map.remove(id.raw().0 as usize).unwrap();
        assert_eq!(epoch, id.raw().1);
        value
    }
}

#[derive(Default)]
pub struct Hub {
    pub instances: Arc<Registry<InstanceHandle, InstanceId>>,
    pub adapters: Arc<Registry<AdapterHandle, AdapterId>>,
    pub devices: Arc<Registry<DeviceHandle, DeviceId>>,
    pub pipeline_layouts: Arc<Registry<PipelineLayoutHandle, PipelineLayoutId>>,
    pub bind_group_layouts: Arc<Registry<BindGroupLayoutHandle, BindGroupLayoutId>>,
    pub bind_groups: Arc<Registry<BindGroupHandle, BindGroupId>>,
    pub shader_modules: Arc<Registry<ShaderModuleHandle, ShaderModuleId>>,
    pub command_buffers: Arc<Registry<CommandBufferHandle, CommandBufferId>>,
    pub render_pipelines: Arc<Registry<RenderPipelineHandle, RenderPipelineId>>,
    pub compute_pipelines: Arc<Registry<ComputePipelineHandle, ComputePipelineId>>,
    pub render_passes: Arc<Registry<RenderPassHandle, RenderPassId>>,
    pub compute_passes: Arc<Registry<ComputePassHandle, ComputePassId>>,
    pub buffers: Arc<Registry<BufferHandle, BufferId>>,
    pub textures: Arc<Registry<TextureHandle, TextureId>>,
    pub texture_views: Arc<Registry<TextureViewHandle, TextureViewId>>,
    pub samplers: Arc<Registry<SamplerHandle, SamplerId>>,
    pub surfaces: Arc<Registry<SurfaceHandle, SurfaceId>>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
