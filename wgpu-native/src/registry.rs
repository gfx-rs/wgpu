#[cfg(feature = "remote")]
use hal::backend::FastHashMap;
#[cfg(feature = "remote")]
use parking_lot::{Mutex, MutexGuard};
#[cfg(not(feature = "remote"))]
use std::marker::PhantomData;
#[cfg(not(feature = "remote"))]
use std::os::raw::c_void;
#[cfg(feature = "remote")]
use std::sync::Arc;

use {
    AdapterHandle, AttachmentStateHandle, BindGroupLayoutHandle, BlendStateHandle,
    CommandBufferHandle, DepthStencilStateHandle, DeviceHandle, InstanceHandle,
    RenderPassHandle, ComputePassHandle,
    PipelineLayoutHandle, RenderPipelineHandle, ShaderModuleHandle,
};

#[cfg(not(feature = "remote"))]
pub type Id = *mut c_void;
#[cfg(feature = "remote")]
pub type Id = u32;

type Item<'a, T> = &'a T;
type ItemMut<'a, T> = &'a mut T;

#[cfg(not(feature = "remote"))]
type ItemsGuard<'a, T> = LocalItems<T>;
#[cfg(feature = "remote")]
type ItemsGuard<'a, T> = MutexGuard<'a, RemoteItems<T>>;

pub trait Registry<T>: Default {
    fn lock(&self) -> ItemsGuard<T>;
}

pub trait Items<T> {
    fn register(&mut self, handle: T) -> Id;
    fn get(&self, id: Id) -> Item<T>;
    fn get_mut(&mut self, id: Id) -> ItemMut<T>;
    fn take(&mut self, id: Id) -> T;
}

#[cfg(not(feature = "remote"))]
pub struct LocalItems<T> {
    marker: PhantomData<T>,
}

#[cfg(not(feature = "remote"))]
impl<T> Items<T> for LocalItems<T> {
    fn register(&mut self, handle: T) -> Id {
        Box::into_raw(Box::new(handle)) as *mut _ as *mut c_void
    }

    fn get(&self, id: Id) -> Item<T> {
        unsafe { (id as *mut T).as_ref() }.unwrap()
    }

    fn get_mut(&mut self, id: Id) -> ItemMut<T> {
        unsafe { (id as *mut T).as_mut() }.unwrap()
    }

    fn take(&mut self, id: Id) -> T {
        unsafe { *Box::from_raw(id as *mut T) }
    }
}

#[cfg(not(feature = "remote"))]
pub struct LocalRegistry<T> {
    marker: PhantomData<T>,
}
#[cfg(not(feature = "remote"))]
impl<T> Default for LocalRegistry<T> {
    fn default() -> Self {
        LocalRegistry {
            marker: PhantomData,
        }
    }
}
#[cfg(not(feature = "remote"))]
impl<T> Registry<T> for LocalRegistry<T> {
    fn lock(&self) -> ItemsGuard<T> {
        LocalItems {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "remote")]
pub struct RemoteItems<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
    free: Vec<Id>,
}

#[cfg(feature = "remote")]
impl<T> RemoteItems<T> {
    fn new() -> Self {
        RemoteItems {
            next_id: 0,
            tracked: FastHashMap::default(),
            free: Vec::new(),
        }
    }
}

#[cfg(feature = "remote")]
impl<T> Items<T> for RemoteItems<T> {
    fn register(&mut self, handle: T) -> Id {
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                self.next_id += 1;
                self.next_id - 1
            }
        };
        self.tracked.insert(id, handle);
        id
    }

    fn get(&self, id: Id) -> Item<T> {
        self.tracked.get(&id).unwrap()
    }

    fn get_mut(&mut self, id: Id) -> ItemMut<T> {
        self.tracked.get_mut(&id).unwrap()
    }

    fn take(&mut self, id: Id) -> T {
        self.free.push(id);
        self.tracked.remove(&id).unwrap()
    }
}

#[cfg(feature = "remote")]
pub struct RemoteRegistry<T> {
    items: Arc<Mutex<RemoteItems<T>>>,
}
#[cfg(feature = "remote")]
impl<T> Default for RemoteRegistry<T> {
    fn default() -> Self {
        RemoteRegistry {
            items: Arc::new(Mutex::new(RemoteItems::new())),
        }
    }
}
#[cfg(feature = "remote")]
impl<T> Registry<T> for RemoteRegistry<T> {
    fn lock(&self) -> ItemsGuard<T> {
        self.items.lock()
    }
}

#[cfg(not(feature = "remote"))]
type ConcreteRegistry<T> = LocalRegistry<T>;
#[cfg(feature = "remote")]
type ConcreteRegistry<T> = RemoteRegistry<T>;

#[derive(Default)]
pub struct Hub {
    pub(crate) instances: ConcreteRegistry<InstanceHandle>,
    pub(crate) adapters: ConcreteRegistry<AdapterHandle>,
    pub(crate) devices: ConcreteRegistry<DeviceHandle>,
    pub(crate) pipeline_layouts: ConcreteRegistry<PipelineLayoutHandle>,
    pub(crate) bind_group_layouts: ConcreteRegistry<BindGroupLayoutHandle>,
    pub(crate) attachment_states: ConcreteRegistry<AttachmentStateHandle>,
    pub(crate) blend_states: ConcreteRegistry<BlendStateHandle>,
    pub(crate) depth_stencil_states: ConcreteRegistry<DepthStencilStateHandle>,
    pub(crate) shader_modules: ConcreteRegistry<ShaderModuleHandle>,
    pub(crate) command_buffers: ConcreteRegistry<CommandBufferHandle>,
    pub(crate) render_pipelines: ConcreteRegistry<RenderPipelineHandle>,
    pub(crate) render_passes: ConcreteRegistry<RenderPassHandle>,
    pub(crate) compute_passes: ConcreteRegistry<ComputePassHandle>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
