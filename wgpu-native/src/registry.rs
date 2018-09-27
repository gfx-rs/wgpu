use std::marker::PhantomData;
use std::os::raw::c_void;
#[cfg(feature = "remote")]
use std::sync::Arc;
#[cfg(feature = "remote")]
use parking_lot::{Mutex, MutexGuard, MappedMutexGuard};
#[cfg(feature = "remote")]
use std::{borrow, cmp, fmt, ops, ptr};

#[cfg(feature = "remote")]
use hal::backend::FastHashMap;

use {AdapterHandle, CommandBufferHandle, DeviceHandle, InstanceHandle, ShaderModuleHandle};


#[cfg(not(feature = "remote"))]
pub(crate) type Id = *mut c_void;
#[cfg(feature = "remote")]
pub(crate) type Id = u32;

#[cfg(not(feature = "remote"))]
type RegistryItem<'a, T> = &'a mut T;
#[cfg(feature = "remote")]
type RegistryItem<'a, T> = MappedMutexGuard<'a, T>;

pub(crate) trait Registry<T> {
    fn new() -> Self;
    fn register(&self, handle: T) -> Id;
    fn get_mut(&self, id: Id) -> RegistryItem<T>;
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

    fn get_mut(&self, id: Id) -> RegistryItem<T> {
        unsafe { (id as *mut T).as_mut() }.unwrap()
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
        let mut registrations = self.registrations.lock();
        let id = registrations.next_id;
        registrations.tracked.insert(id, handle);
        registrations.next_id += 1;
        id
    }

    fn get_mut(&self, id: Id) -> RegistryItem<T> {
        MutexGuard::map(self.registrations.lock(), |r| r.tracked.get_mut(&id).unwrap())
    }
}

#[cfg(not(feature = "remote"))]
type ConcreteRegistry<T> = LocalRegistry<T>;
#[cfg(feature = "remote")]
type ConcreteRegistry<T> = RemoteRegistry<T>;

lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: ConcreteRegistry<AdapterHandle> = ConcreteRegistry::new();
    pub(crate) static ref DEVICE_REGISTRY: ConcreteRegistry<DeviceHandle> = ConcreteRegistry::new();
    pub(crate) static ref INSTANCE_REGISTRY: ConcreteRegistry<InstanceHandle> = ConcreteRegistry::new();
    pub(crate) static ref SHADER_MODULE_REGISTRY: ConcreteRegistry<ShaderModuleHandle> = ConcreteRegistry::new();
    pub(crate) static ref COMMAND_BUFFER_REGISTRY: ConcreteRegistry<CommandBufferHandle> = ConcreteRegistry::new();
}
