// Box casts are needed, alternative would be a temporaries which are more verbose and not more expressive.
#![allow(trivial_casts)]

use crate::{Capabilities, Instance, InstanceError};

use super::{DynAdapter, DynResource, DynResourceExt as _, DynSurface};

pub struct DynExposedAdapter {
    pub adapter: Box<dyn DynAdapter>,
    pub info: wgt::AdapterInfo,
    pub features: wgt::Features,
    pub capabilities: Capabilities,
}

pub trait DynInstance: DynResource {
    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Box<dyn DynSurface>, InstanceError>;

    unsafe fn enumerate_adapters(
        &self,
        surface_hint: Option<&dyn DynSurface>,
    ) -> Vec<DynExposedAdapter>;
}

impl<I: Instance + DynResource> DynInstance for I {
    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Box<dyn DynSurface>, InstanceError> {
        unsafe { I::create_surface(self, display_handle, window_handle) }
            .map(|surface| Box::new(surface) as Box<dyn DynSurface>)
    }

    unsafe fn enumerate_adapters(
        &self,
        surface_hint: Option<&dyn DynSurface>,
    ) -> Vec<DynExposedAdapter> {
        let surface_hint = surface_hint.map(|s| s.expect_downcast_ref());
        unsafe { I::enumerate_adapters(self, surface_hint) }
            .into_iter()
            .map(|exposed| DynExposedAdapter {
                adapter: Box::new(exposed.adapter),
                info: exposed.info,
                features: exposed.features,
                capabilities: exposed.capabilities,
            })
            .collect()
    }
}
