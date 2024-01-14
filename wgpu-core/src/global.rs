use std::{marker::PhantomData, sync::Arc};

use wgt::Backend;

use crate::{
    hal_api::HalApi,
    hub::{HubReport, Hubs},
    id::SurfaceId,
    identity::GlobalIdentityHandlerFactory,
    instance::{Instance, Surface},
    registry::{Registry, RegistryReport},
    resource_log,
    storage::Element,
};

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalReport {
    pub surfaces: RegistryReport,
    #[cfg(vulkan)]
    pub vulkan: Option<HubReport>,
    #[cfg(metal)]
    pub metal: Option<HubReport>,
    #[cfg(dx12)]
    pub dx12: Option<HubReport>,
    #[cfg(gles)]
    pub gl: Option<HubReport>,
}

impl GlobalReport {
    pub fn surfaces(&self) -> &RegistryReport {
        &self.surfaces
    }
    pub fn hub_report(&self, backend: Backend) -> &HubReport {
        match backend {
            #[cfg(vulkan)]
            Backend::Vulkan => self.vulkan.as_ref().unwrap(),
            #[cfg(metal)]
            Backend::Metal => self.metal.as_ref().unwrap(),
            #[cfg(dx12)]
            Backend::Dx12 => self.dx12.as_ref().unwrap(),
            #[cfg(gles)]
            Backend::Gl => self.gl.as_ref().unwrap(),
            _ => panic!("HubReport is not supported on this backend"),
        }
    }
}

pub struct Global<G: GlobalIdentityHandlerFactory> {
    pub instance: Instance,
    pub surfaces: Registry<SurfaceId, Surface>,
    pub(crate) hubs: Hubs,
    _phantom: PhantomData<G>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn new(name: &str, factory: G, instance_desc: wgt::InstanceDescriptor) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc),
            surfaces: Registry::without_backend(&factory),
            hubs: Hubs::new(&factory),
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: HalApi>(
        name: &str,
        factory: G,
        hal_instance: A::Instance,
    ) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: A::create_instance_from_hal(name, hal_instance),
            surfaces: Registry::without_backend(&factory),
            hubs: Hubs::new(&factory),
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: HalApi>(&self) -> Option<&A::Instance> {
        A::instance_as_hal(&self.instance)
    }

    /// # Safety
    ///
    /// - The raw handles obtained from the Instance must not be manually destroyed
    pub unsafe fn from_instance(factory: G, instance: Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance,
            surfaces: Registry::without_backend(&factory),
            hubs: Hubs::new(&factory),
            _phantom: PhantomData,
        }
    }

    pub fn clear_backend<A: HalApi>(&self, _dummy: ()) {
        let hub = A::hub(self);
        let surfaces_locked = self.surfaces.read();
        // this is used for tests, which keep the adapter
        hub.clear(&surfaces_locked, false);
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            #[cfg(vulkan)]
            vulkan: if self.instance.vulkan.is_some() {
                Some(self.hubs.vulkan.generate_report())
            } else {
                None
            },
            #[cfg(metal)]
            metal: if self.instance.metal.is_some() {
                Some(self.hubs.metal.generate_report())
            } else {
                None
            },
            #[cfg(dx12)]
            dx12: if self.instance.dx12.is_some() {
                Some(self.hubs.dx12.generate_report())
            } else {
                None
            },
            #[cfg(gles)]
            gl: if self.instance.gl.is_some() {
                Some(self.hubs.gl.generate_report())
            } else {
                None
            },
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Drop for Global<G> {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        resource_log!("Global::drop");
        let mut surfaces_locked = self.surfaces.write();

        // destroy hubs before the instance gets dropped
        #[cfg(vulkan)]
        {
            self.hubs.vulkan.clear(&surfaces_locked, true);
        }
        #[cfg(metal)]
        {
            self.hubs.metal.clear(&surfaces_locked, true);
        }
        #[cfg(dx12)]
        {
            self.hubs.dx12.clear(&surfaces_locked, true);
        }
        #[cfg(gles)]
        {
            self.hubs.gl.clear(&surfaces_locked, true);
        }

        // destroy surfaces
        for element in surfaces_locked.map.drain(..) {
            if let Element::Occupied(arc_surface, _) = element {
                if let Some(surface) = Arc::into_inner(arc_surface) {
                    self.instance.destroy_surface(surface);
                } else {
                    panic!("Surface cannot be destroyed because is still in use");
                }
            }
        }
    }
}

#[cfg(send_sync)]
fn _test_send_sync(global: &Global<crate::identity::IdentityManagerFactory>) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
