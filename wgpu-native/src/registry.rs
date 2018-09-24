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
    fn register(&self, handle: T) -> Id;
    fn get(&self, id: Id) -> Option<&T>;
    fn get_mut(&self, id: Id) -> Option<&mut T>;
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
        ::std::boxed::Box::into_raw(Box::new(handle)) as *mut _ as *mut c_void
    }

    fn get(&self, id: Id) -> Option<&T> {
        unsafe { (id as *const T).as_ref() }
    }

    fn get_mut(&self, id: Id) -> Option<&mut T> {
        unsafe { (id as *mut T).as_mut() }
    }
}

#[cfg(feature = "remote")]
struct Registrations<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
}

#[cfg(feature = "remote")]
impl<T> Registrations<T> {
    fn new() -> Self {
        Registrations {
            next_id: 0,
            tracked: FastHashMap::default(),
        }
    }
}

#[cfg(feature = "remote")]
pub(crate) struct RemoteRegistry<T> {
    registrations: Arc<Mutex<Registrations<T>>>,
}

#[cfg(feature = "remote")]
impl<T> Registry<T> for RemoteRegistry<T> {
    fn new() -> Self {
        RemoteRegistry {
            registrations: Arc::new(Mutex::new(Registrations::new())),
        }
    }

    fn register(&self, handle: T) -> Id {
        let mut registrations = self.registrations.lock().unwrap();
        let id = registrations.next_id;
        registrations.tracked.insert(id, handle);
        registrations.next_id += 1;
        id
    }

    fn get(&self, id: Id) -> Option<&T> {
        let registrations = self.registrations.lock().unwrap();
        registrations.tracked.get(&id)
    }

    fn get_mut(&self, id: Id) -> Option<&mut T> {
        let registrations = self.registrations.lock().unwrap();
        registrations.tracked.get_mut(&id)
    }
}

#[cfg(not(feature = "remote"))]
lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: LocalRegistry<AdapterHandle> = LocalRegistry::new();
    pub(crate) static ref DEVICE_REGISTRY: LocalRegistry<DeviceHandle> = LocalRegistry::new();
    pub(crate) static ref INSTANCE_REGISTRY: LocalRegistry<InstanceHandle> = LocalRegistry::new();
    pub(crate) static ref SHADER_MODULE_REGISTRY: LocalRegistry<ShaderModuleHandle> = LocalRegistry::new();
}

#[cfg(feature = "remote")]
lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: RemoteRegistry<AdapterHandle> = RemoteRegistry::new();
    pub(crate) static ref DEVICE_REGISTRY: RemoteRegistry<DeviceHandle> = RemoteRegistry::new();
    pub(crate) static ref INSTANCE_REGISTRY: RemoteRegistry<InstanceHandle> = RemoteRegistry::new();
    pub(crate) static ref SHADER_MODULE_REGISTRY: RemoteRegistry<ShaderModuleHandle> = RemoteRegistry::new();
}
