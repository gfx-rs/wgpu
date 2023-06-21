use crate::{
    hal_api::HalApi,
    hub::{HubReport, Hubs},
    id,
    identity::GlobalIdentityHandlerFactory,
    instance::{Instance, Surface},
    registry::Registry,
    storage::{Element, StorageReport},
};

#[derive(Debug)]
pub struct GlobalReport {
    pub surfaces: StorageReport,
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    pub vulkan: Option<HubReport>,
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub metal: Option<HubReport>,
    #[cfg(all(feature = "dx12", windows))]
    pub dx12: Option<HubReport>,
    #[cfg(all(feature = "dx11", windows))]
    pub dx11: Option<HubReport>,
    #[cfg(feature = "gles")]
    pub gl: Option<HubReport>,
}

pub struct Global<G: GlobalIdentityHandlerFactory> {
    pub instance: Instance,
    pub surfaces: Registry<Surface, id::SurfaceId, G>,
    pub(crate) hubs: Hubs<G>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn new(name: &str, factory: G, instance_desc: wgt::InstanceDescriptor) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc),
            surfaces: Registry::without_backend(&factory, "Surface"),
            hubs: Hubs::new(&factory),
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
            surfaces: Registry::without_backend(&factory, "Surface"),
            hubs: Hubs::new(&factory),
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
            surfaces: Registry::without_backend(&factory, "Surface"),
            hubs: Hubs::new(&factory),
        }
    }

    pub fn clear_backend<A: HalApi>(&self, _dummy: ()) {
        let mut surface_guard = self.surfaces.data.write();
        let hub = A::hub(self);
        // this is used for tests, which keep the adapter
        hub.clear(&mut surface_guard, false);
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.data.read().generate_report(),
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: if self.instance.vulkan.is_some() {
                Some(self.hubs.vulkan.generate_report())
            } else {
                None
            },
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            metal: if self.instance.metal.is_some() {
                Some(self.hubs.metal.generate_report())
            } else {
                None
            },
            #[cfg(all(feature = "dx12", windows))]
            dx12: if self.instance.dx12.is_some() {
                Some(self.hubs.dx12.generate_report())
            } else {
                None
            },
            #[cfg(all(feature = "dx11", windows))]
            dx11: if self.instance.dx11.is_some() {
                Some(self.hubs.dx11.generate_report())
            } else {
                None
            },
            #[cfg(feature = "gles")]
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
        log::info!("Dropping Global");
        let mut surface_guard = self.surfaces.data.write();

        // destroy hubs before the instance gets dropped
        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        {
            self.hubs.vulkan.clear(&mut surface_guard, true);
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            self.hubs.metal.clear(&mut surface_guard, true);
        }
        #[cfg(all(feature = "dx12", windows))]
        {
            self.hubs.dx12.clear(&mut surface_guard, true);
        }
        #[cfg(all(feature = "dx11", windows))]
        {
            self.hubs.dx11.clear(&mut surface_guard, true);
        }
        #[cfg(feature = "gles")]
        {
            self.hubs.gl.clear(&mut surface_guard, true);
        }

        // destroy surfaces
        for element in surface_guard.map.drain(..) {
            if let Element::Occupied(surface, _) = element {
                self.instance.destroy_surface(surface);
            }
        }
    }
}

#[cfg(all(
    test,
    any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )
))]
fn _test_send_sync(global: &Global<crate::identity::IdentityManagerFactory>) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
