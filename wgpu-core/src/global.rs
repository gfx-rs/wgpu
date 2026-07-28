use alloc::{borrow::ToOwned as _, sync::Arc};
use core::fmt;

use crate::{
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
    pub fn new(
        name: &str,
        instance_desc: wgt::InstanceDescriptor,
        telemetry: Option<hal::Telemetry>,
    ) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc, telemetry),
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: hal::Api>(name: &str, hal_instance: A::Instance) -> Self {
        profiling::scope!("Global::new");

        Self {
            instance: Instance::from_hal_instance::<A>(name.to_owned(), hal_instance),
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: hal::Api>(&self) -> Option<&A::Instance> {
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

/// Implement [`Send`] + [`Sync`] for [`Global`], and check that all of its fields are.
///
/// This is identical to the “auto trait” implementation that Rust would provide, except that
/// it is eager rather than lazy: its requirements are checked now (in
/// `_global_fields_are_send_sync`), when this crate is compiled, rather than whenever a dependent
/// wants to know whether `Global: Send` holds.
///
/// This improves compilation performance and avoids a risk of dependents running into the default
/// [`recursion_limit`] when checking types containing [`Global`]. This risk will become greater
/// when Rust’s “next solver” is stabilized.
///
/// [`recursion_limit`]: https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute
#[cfg(send_sync)]
mod global_send_sync {
    use super::Global;

    // SAFETY: Bounds checked below
    unsafe impl Send for Global {}

    // SAFETY: Bounds checked below
    unsafe impl Sync for Global {}

    /// This function will fail to compile if any field is not `Send + Sync`, or if a new field is
    /// added to `Global`.
    ///
    /// This technique is modeled after the macro library `non_structural_derive`, with permission
    /// (see
    /// <https://github.com/fee1-dead/non_structural_derive/issues/1#issuecomment-5250905440>).
    /// We only need it once, so it’s cheaper to write out the same code the macro would generate.
    fn _global_fields_are_send_sync(global: &Global) {
        fn _check_bound<T: Send + Sync>(_: &T) {}

        let Global {
            surfaces,
            hub,
            instance,
        } = global;
        _check_bound(surfaces);
        _check_bound(hub);
        _check_bound(instance);
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
