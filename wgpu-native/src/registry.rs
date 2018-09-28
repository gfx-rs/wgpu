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
    AdapterHandle, AttachmentStateHandle, BindGroupLayoutHandle, BlendStateHandle, CommandBufferHandle,
    DepthStencilStateHandle, DeviceHandle, InstanceHandle, PipelineLayoutHandle,
    RenderPipelineHandle, ShaderModuleHandle,
};

#[cfg(not(feature = "remote"))]
pub(crate) type Id = *mut c_void;
#[cfg(feature = "remote")]
pub(crate) type Id = u32;

type Item<'a, T> = &'a T;
type ItemMut<'a, T> = &'a mut T;

#[cfg(not(feature = "remote"))]
type ItemsGuard<'a, T> = LocalItems<T>;
#[cfg(feature = "remote")]
type ItemsGuard<'a, T> = MutexGuard<'a, RemoteItems<T>>;

pub(crate) trait Registry<T> {
    fn new() -> Self;
    fn register(&self, handle: T) -> Id;
    fn lock(&self) -> ItemsGuard<T>;
}

pub(crate) trait Items<T> {
    fn get(&self, id: Id) -> Item<T>;
    fn get_mut(&mut self, id: Id) -> ItemMut<T>;
    fn take(&self, id: Id) -> T;
}

#[cfg(not(feature = "remote"))]
pub(crate) struct LocalItems<T> {
    marker: PhantomData<T>,
}

#[cfg(not(feature = "remote"))]
impl<T> Items<T> for LocalItems<T> {
    fn get(&self, id: Id) -> Item<T> {
        unsafe { (id as *mut T).as_ref() }.unwrap()
    }

    fn get_mut(&mut self, id: Id) -> ItemMut<T> {
        unsafe { (id as *mut T).as_mut() }.unwrap()
    }
}

#[cfg(not(feature = "remote"))]
pub(crate) struct LocalRegistry<T> {
    marker: PhantomData<T>,
}

#[cfg(not(feature = "remote"))]
impl<T> Registry<T> for LocalRegistry<T> {
    fn new() -> Self {
        LocalRegistry {
            marker: PhantomData,
        }
    }

    fn register(&self, handle: T) -> Id {
        Box::into_raw(Box::new(handle)) as *mut _ as *mut c_void
    }

    fn lock(&self) -> ItemsGuard<T> {
        LocalItems {
            marker: PhantomData,
        }
    }

    fn take(&self, id: Id) -> T {
        unsafe {
            *Box::from_raw(id as *mut T)
        }
    }
}

#[cfg(feature = "remote")]
pub(crate) struct RemoteItems<T> {
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
    fn get(&self, id: Id) -> Item<T> {
        self.tracked.get(&id).unwrap()
    }

    fn get_mut(&mut self, id: Id) -> ItemMut<T> {
        self.tracked.get_mut(&id).unwrap()
    }
}

#[cfg(feature = "remote")]
pub(crate) struct RemoteRegistry<T> {
    items: Arc<Mutex<RemoteItems<T>>>,
}

#[cfg(feature = "remote")]
impl<T> Registry<T> for RemoteRegistry<T> {
    fn new() -> Self {
        RemoteRegistry {
            items: Arc::new(Mutex::new(RemoteItems::new())),
        }
    }

    fn register(&self, handle: T) -> Id {
        let mut items = self.items.lock();
        let id = match items.free.pop() {
            Some(id) => id,
            None => {
                items.next_id += 1;
                items.next_id - 1
            }
        };
        items.tracked.insert(id, handle);
        id
    }

    fn lock(&self) -> ItemsGuard<T> {
        self.items.lock()
    }

    fn take(&self, id: Id) -> T {
        let mut registrations = self.registrations.lock();
        registrations.free.push(id);
        registrations.tracked.remove(&id).unwrap()
    }
}

#[cfg(not(feature = "remote"))]
type ConcreteRegistry<T> = LocalRegistry<T>;
#[cfg(feature = "remote")]
type ConcreteRegistry<T> = RemoteRegistry<T>;

lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: ConcreteRegistry<AdapterHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref ATTACHMENT_STATE_REGISTRY: ConcreteRegistry<AttachmentStateHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref BIND_GROUP_LAYOUT_REGISTRY: ConcreteRegistry<BindGroupLayoutHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref BLEND_STATE_REGISTRY: ConcreteRegistry<BlendStateHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref DEPTH_STENCIL_STATE_REGISTRY: ConcreteRegistry<DepthStencilStateHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref DEVICE_REGISTRY: ConcreteRegistry<DeviceHandle> = ConcreteRegistry::new();
    pub(crate) static ref COMMAND_BUFFER_REGISTRY: ConcreteRegistry<CommandBufferHandle> = ConcreteRegistry::new();
    pub(crate) static ref INSTANCE_REGISTRY: ConcreteRegistry<InstanceHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref PIPELINE_LAYOUT_REGISTRY: ConcreteRegistry<PipelineLayoutHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref RENDER_PIPELINE_REGISTRY: ConcreteRegistry<RenderPipelineHandle> =
        ConcreteRegistry::new();
    pub(crate) static ref SHADER_MODULE_REGISTRY: ConcreteRegistry<ShaderModuleHandle> =
        ConcreteRegistry::new();
}
