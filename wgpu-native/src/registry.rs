use std::sync::{Arc, Mutex};
use std::{borrow, cmp, fmt, ops, ptr};

use hal::backend::FastHashMap;
use {AdapterHandle, DeviceHandle, InstanceHandle, ShaderModuleHandle};

pub(crate) type Id = u32;

pub(crate) struct Registry<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
}

impl<T> Registry<T> {
    fn new() -> Self {
        Registry {
            next_id: 0,
            tracked: FastHashMap::default(),
        }
    }

    pub(crate) fn register(&mut self, handle: T) -> Id {
        let id = self.next_id;
        self.tracked.insert(id, handle);
        self.next_id += 1;
        id
    }

    pub(crate) fn get(&self, id: Id) -> Option<&T> {
        self.tracked.get(&id)
    }

    pub(crate) fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.tracked.get_mut(&id)
    }
}

lazy_static! {
    pub(crate) static ref ADAPTER_REGISTRY: Arc<Mutex<Registry<AdapterHandle>>> =
        Arc::new(Mutex::new(Registry::new()));
    pub(crate) static ref DEVICE_REGISTRY: Arc<Mutex<Registry<DeviceHandle>>> =
        Arc::new(Mutex::new(Registry::new()));
    pub(crate) static ref INSTANCE_REGISTRY: Arc<Mutex<Registry<InstanceHandle>>> =
        Arc::new(Mutex::new(Registry::new()));
    pub(crate) static ref SHADER_MODULE_REGISTRY: Arc<Mutex<Registry<ShaderModuleHandle>>> =
        Arc::new(Mutex::new(Registry::new()));
}
