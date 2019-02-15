use crate::{
    AdapterHandle, BindGroupLayoutHandle, BindGroupHandle,
    CommandBufferHandle, DeviceHandle, InstanceHandle,
    RenderPassHandle, ComputePassHandle,
    PipelineLayoutHandle, RenderPipelineHandle, ComputePipelineHandle, ShaderModuleHandle,
    BufferHandle, SamplerHandle, TextureHandle, TextureViewHandle,
    SurfaceHandle, SwapChainHandle,
};

use hal::backend::FastHashMap;
use lazy_static::lazy_static;
use parking_lot::RwLock;
#[cfg(feature = "local")]
use parking_lot::Mutex;

use std::ops;
use std::sync::Arc;

//TODO: use Vec instead of HashMap
//TODO: track epochs of indices

pub type Id = u32;

/// A simple structure to manage identities of objects.
#[derive(Default)]
pub struct IdentityManager {
    last_id: Id,
    free: Vec<Id>,
}

impl IdentityManager {
    pub fn alloc(&mut self) -> Id {
        match self.free.pop() {
            Some(id) => id,
            None => {
                self.last_id += 1;
                assert_ne!(self.last_id, 0);
                self.last_id
            }
        }
    }

    pub fn free(&mut self, id: Id) {
        debug_assert!(id <= self.last_id && !self.free.contains(&id));
        self.free.push(id);
    }
}

pub struct Storage<T> {
    //TODO: consider concurrent hashmap?
    map: FastHashMap<Id, T>,
}

impl<T> Storage<T> {
    pub fn get(&self, id: Id) -> &T {
        self.map.get(&id).unwrap()
    }
    pub fn get_mut(&mut self, id: Id) -> &mut T {
        self.map.get_mut(&id).unwrap()
    }
    pub fn take(&mut self, id: Id) -> T {
        self.map.remove(&id).unwrap()
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
                map: FastHashMap::default(),
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

#[cfg(feature = "local")]
impl<T> Registry<T> {
    pub fn register(&self, value: T) -> Id {
        let id = self.identity.lock().alloc();
        let old = self.data.write().map.insert(id, value);
        assert!(old.is_none());
        id
    }
    pub fn unregister(&self, id: Id) {
        self.identity.lock().free(id);
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
    pub(crate) swap_chains: Arc<Registry<SwapChainHandle>>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
