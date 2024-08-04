use std::collections::HashMap;

use wgt::Backend;

use crate::{
    hal_api::HalApi,
    hub::{HubReport, Hubs},
    instance::{Instance, Surface},
    registry::{Registry, RegistryReport},
    resource_log,
};

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalReport {
    pub surfaces: RegistryReport,
    pub report_per_backend: HashMap<Backend, HubReport>,
}

impl GlobalReport {
    pub fn surfaces(&self) -> &RegistryReport {
        &self.surfaces
    }
    pub fn hub_report(&self, backend: Backend) -> &HubReport {
        self.report_per_backend.get(&backend).unwrap()
    }
}

pub struct Global {
    pub instance: Instance,
    pub(crate) surfaces: Registry<Surface>,
    pub(crate) hubs: Hubs,
}

impl Global {
    pub fn new(name: &str, instance_desc: wgt::InstanceDescriptor) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc),
            surfaces: Registry::without_backend(),
            hubs: Hubs::new(),
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
            surfaces: Registry::without_backend(),
            hubs: Hubs::new(),
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: HalApi>(&self) -> Option<&A::Instance> {
        self.instance.raw(A::VARIANT).map(|instance| {
            instance
                .as_any()
                .downcast_ref()
                // This should be impossible. It would mean that backend instance and enum type are mismatching.
                .expect("Stored instance is not of the correct type")
        })
    }

    /// # Safety
    ///
    /// - The raw handles obtained from the Instance must not be manually destroyed
    pub unsafe fn from_instance(instance: Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance,
            surfaces: Registry::without_backend(),
            hubs: Hubs::new(),
        }
    }

    pub fn generate_report(&self) -> GlobalReport {
        let mut report_per_backend = HashMap::default();
        let instance_per_backend = &self.instance.instance_per_backend;

        #[cfg(vulkan)]
        if instance_per_backend
            .iter()
            .any(|(backend, _)| backend == &Backend::Vulkan)
        {
            report_per_backend.insert(Backend::Vulkan, self.hubs.vulkan.generate_report());
        };
        #[cfg(metal)]
        if instance_per_backend
            .iter()
            .any(|(backend, _)| backend == &Backend::Metal)
        {
            report_per_backend.insert(Backend::Metal, self.hubs.metal.generate_report());
        };
        #[cfg(dx12)]
        if instance_per_backend
            .iter()
            .any(|(backend, _)| backend == &Backend::Dx12)
        {
            report_per_backend.insert(Backend::Dx12, self.hubs.dx12.generate_report());
        };
        #[cfg(gles)]
        if instance_per_backend
            .iter()
            .any(|(backend, _)| backend == &Backend::Gl)
        {
            report_per_backend.insert(Backend::Gl, self.hubs.gl.generate_report());
        };

        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            report_per_backend,
        }
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        resource_log!("Global::drop");
        let mut surfaces_locked = self.surfaces.write();

        // destroy hubs before the instance gets dropped
        #[cfg(vulkan)]
        {
            self.hubs.vulkan.clear(&surfaces_locked);
        }
        #[cfg(metal)]
        {
            self.hubs.metal.clear(&surfaces_locked);
        }
        #[cfg(dx12)]
        {
            self.hubs.dx12.clear(&surfaces_locked);
        }
        #[cfg(gles)]
        {
            self.hubs.gl.clear(&surfaces_locked);
        }

        surfaces_locked.map.clear();
    }
}

#[cfg(send_sync)]
fn _test_send_sync(global: &Global) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
