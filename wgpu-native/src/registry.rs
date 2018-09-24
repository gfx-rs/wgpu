use std::marker::PhantomData;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};
use std::{borrow, cmp, fmt, ops, ptr};

use hal::backend::FastHashMap;
use {AdapterHandle, DeviceHandle, InstanceHandle, ShaderModuleHandle};

#[cfg(not(feature = "remote"))]
pub(crate) type Id = *mut c_void;
#[cfg(feature = "remote")]
pub(crate) type Id = u32;

pub(crate) trait Registry<T> {
    fn new() -> Self;
    fn register(&mut self, handle: T) -> Id;
    fn get(&self, id: Id) -> Option<&T>;
    fn get_mut(&mut self, id: Id) -> Option<&mut T>;
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

    fn register(&mut self, handle: T) -> Id {
        ::std::boxed::Box::into_raw(Box::new(handle)) as *mut _ as *mut c_void
    }

    fn get(&self, id: Id) -> Option<&T> {
        unsafe { (id as *const T).as_ref() }
    }

    fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        unsafe { (id as *mut T).as_mut() }
    }
}

#[cfg(feature = "remote")]
pub(crate) struct RemoteRegistry<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
}

#[cfg(feature = "remote")]
impl<T> Registry<T> for RemoteRegistry<T> {
    fn new() -> Self {
        RemoteRegistry {
            next_id: 0,
            tracked: FastHashMap::default(),
        }
    }

    fn register(&mut self, handle: T) -> Id {
        let id = self.next_id;
        self.tracked.insert(id, handle);
        self.next_id += 1;
        id
    }

    fn get(&self, id: Id) -> Option<&T> {
        self.tracked.get(&id)
    }

    fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.tracked.get_mut(&id)
    }
}

#[cfg(not(feature = "remote"))]
lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: Mutex<LocalRegistry<AdapterHandle>> =
        Mutex::new(LocalRegistry::new());
    pub(crate) static ref DEVICE_REGISTRY: Mutex<LocalRegistry<DeviceHandle>> =
        Mutex::new(LocalRegistry::new());
    pub(crate) static ref INSTANCE_REGISTRY: Mutex<LocalRegistry<InstanceHandle>> =
        Mutex::new(LocalRegistry::new());
    pub(crate) static ref SHADER_MODULE_REGISTRY: Mutex<LocalRegistry<ShaderModuleHandle>> =
        Mutex::new(LocalRegistry::new());
}

#[cfg(feature = "remote")]
lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: Arc<Mutex<RemoteRegistry<AdapterHandle>>> =
        Arc::new(Mutex::new(RemoteRegistry::new()));
    pub(crate) static ref DEVICE_REGISTRY: Arc<Mutex<RemoteRegistry<DeviceHandle>>> =
        Arc::new(Mutex::new(RemoteRegistry::new()));
    pub(crate) static ref INSTANCE_REGISTRY: Arc<Mutex<RemoteRegistry<InstanceHandle>>> =
        Arc::new(Mutex::new(RemoteRegistry::new()));
    pub(crate) static ref SHADER_MODULE_REGISTRY: Arc<Mutex<RemoteRegistry<ShaderModuleHandle>>> =
        Arc::new(Mutex::new(RemoteRegistry::new()));
}
