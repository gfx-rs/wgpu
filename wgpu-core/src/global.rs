use std::{fmt, sync::Arc};

use crate::{
    hal_api::HalApi,
    hub::{Hub, HubReport},
    instance::{Instance, Surface},
    registry::{Registry, RegistryReport},
    resource_log,
};

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalReport {
    pub surfaces: RegistryReport,
    pub hub: HubReport,
}

impl GlobalReport {
    pub fn surfaces(&self) -> &RegistryReport {
        &self.surfaces
    }
    pub fn hub_report(&self) -> &HubReport {
        &self.hub
    }
}

pub struct Global {
    pub(crate) surfaces: Registry<Arc<Surface>>,
    pub(crate) hub: Hub,
    // the instance must be dropped last
    pub instance: Instance,
}

impl Global {
    pub fn new(name: &str, instance_desc: wgt::InstanceDescriptor) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc),
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: HalApi>(name: &str, hal_instance: A::Instance) -> Self {
        profiling::scope!("Global::new");

        let dyn_instance: Box<dyn hal::DynInstance> = Box::new(hal_instance);
        Self {
            instance: Instance {
                name: name.to_owned(),
                instance_per_backend: std::iter::once((A::VARIANT, dyn_instance)).collect(),
                ..Default::default()
            },
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: HalApi>(&self) -> Option<&A::Instance> {
        unsafe { self.instance.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw handles obtained from the Instance must not be manually destroyed
    pub unsafe fn from_instance(instance: Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance,
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            hub: self.hub.generate_report(),
        }
    }
}

impl fmt::Debug for Global {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Global").finish()
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        resource_log!("Global::drop");
    }
}

#[cfg(send_sync)]
fn _test_send_sync(global: &Global) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
