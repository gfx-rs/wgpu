use crate::{
    AdapterHandle, BindGroupLayoutHandle, BindGroupHandle,
    CommandBufferHandle, DeviceHandle, InstanceHandle,
    RenderPassHandle, ComputePassHandle,
    PipelineLayoutHandle, RenderPipelineHandle, ComputePipelineHandle, ShaderModuleHandle,
    BufferHandle, SamplerHandle, TextureHandle, TextureViewHandle,
    SurfaceHandle,
};

use lazy_static::lazy_static;
use parking_lot::RwLock;
#[cfg(feature = "local")]
use parking_lot::Mutex;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
use vec_map::VecMap;

use std::ops;
use std::sync::Arc;


pub(crate) type Index = u32;
pub(crate) type Epoch = u32;
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
            Some(index) => {
                Id(index, self.epochs[index as usize])
            }
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

pub struct Storage<T> {
    //TODO: consider concurrent hashmap?
    map: VecMap<(T, Epoch)>,
}

impl<T> ops::Index<Id> for Storage<T> {
    type Output = T;
    fn index(&self, id: Id) -> &T {
        let (ref value, epoch) = self.map[id.0 as usize];
        assert_eq!(epoch, id.1);
        value
    }
}

impl<T> ops::IndexMut<Id> for Storage<T> {
    fn index_mut(&mut self, id: Id) -> &mut T {
        let (ref mut value, epoch) = self.map[id.0 as usize];
        assert_eq!(epoch, id.1);
        value
    }
}

pub struct Registry<T> {
    #[cfg(feature = "local")]
    identity: Mutex<IdentityManager>,
    data: RwLock<Storage<T>>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            #[cfg(feature = "local")]
            identity: Mutex::new(IdentityManager::default()),
            data: RwLock::new(Storage {
                map: VecMap::new(),
            }),
        }
    }
}

impl<T> ops::Deref for Registry<T> {
    type Target = RwLock<Storage<T>>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> ops::DerefMut for Registry<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Registry<T> {
    pub fn register(&self, id: Id, value: T) {
        let old = self.data.write().map.insert(id.0 as usize, (value, id.1));
        assert!(old.is_none());
    }
}

impl<T> Registry<T> {
    #[cfg(feature = "local")]
    pub fn register_local(&self, value: T) -> Id {
        let id = self.identity.lock().alloc();
        self.register(id, value);
        id
    }

    pub fn unregister(&self, id: Id) -> T {
        #[cfg(feature = "local")]
        self.identity.lock().free(id);
        let (value, epoch) = self.data.write().map.remove(id.0 as usize).unwrap();
        assert_eq!(epoch, id.1);
        value
    }
}

#[derive(Default)]
pub struct Hub {
    pub(crate) instances: Arc<Registry<InstanceHandle>>,
    pub(crate) adapters: Arc<Registry<AdapterHandle>>,
    pub(crate) devices: Arc<Registry<DeviceHandle>>,
    pub(crate) pipeline_layouts: Arc<Registry<PipelineLayoutHandle>>,
    pub(crate) bind_group_layouts: Arc<Registry<BindGroupLayoutHandle>>,
    pub(crate) bind_groups: Arc<Registry<BindGroupHandle>>,
    pub(crate) shader_modules: Arc<Registry<ShaderModuleHandle>>,
    pub(crate) command_buffers: Arc<Registry<CommandBufferHandle>>,
    pub(crate) render_pipelines: Arc<Registry<RenderPipelineHandle>>,
    pub(crate) compute_pipelines: Arc<Registry<ComputePipelineHandle>>,
    pub(crate) render_passes: Arc<Registry<RenderPassHandle>>,
    pub(crate) compute_passes: Arc<Registry<ComputePassHandle>>,
    pub(crate) buffers: Arc<Registry<BufferHandle>>,
    pub(crate) textures: Arc<Registry<TextureHandle>>,
    pub(crate) texture_views: Arc<Registry<TextureViewHandle>>,
    pub(crate) samplers: Arc<Registry<SamplerHandle>>,
    pub(crate) surfaces: Arc<Registry<SurfaceHandle>>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
