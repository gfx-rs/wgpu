use wgt::{Backend, WasmNotSendSync};

use crate::{
    global::Global,
    hub::Hub,
    identity::GlobalIdentityHandlerFactory,
    instance::{HalSurface, Instance, Surface},
};

pub trait HalApi: hal::Api + 'static + WasmNotSendSync {
    const VARIANT: Backend;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance;
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance>;
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self>;
    fn get_surface(surface: &Surface) -> Option<&HalSurface<Self>>;
}

impl HalApi for hal::api::Empty {
    const VARIANT: Backend = Backend::Empty;
    fn create_instance_from_hal(_: &str, _: Self::Instance) -> Instance {
        unimplemented!("called empty api")
    }
    fn instance_as_hal(_: &Instance) -> Option<&Self::Instance> {
        unimplemented!("called empty api")
    }
    fn hub<G: GlobalIdentityHandlerFactory>(_: &Global<G>) -> &Hub<Self> {
        unimplemented!("called empty api")
    }
    fn get_surface(_: &Surface) -> Option<&HalSurface<Self>> {
        unimplemented!("called empty api")
    }
}

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
impl HalApi for hal::api::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            vulkan: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.vulkan.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self> {
        &global.hubs.vulkan
    }
    fn get_surface(surface: &Surface) -> Option<&HalSurface<Self>> {
        surface.raw.downcast_ref()
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl HalApi for hal::api::Metal {
    const VARIANT: Backend = Backend::Metal;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            metal: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.metal.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self> {
        &global.hubs.metal
    }
    fn get_surface(surface: &Surface) -> Option<&HalSurface<Self>> {
        surface.raw.downcast_ref()
    }
}

#[cfg(all(feature = "dx12", windows))]
impl HalApi for hal::api::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        Instance {
            name: name.to_owned(),
            dx12: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.dx12.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self> {
        &global.hubs.dx12
    }
    fn get_surface(surface: &Surface) -> Option<&HalSurface<Self>> {
        surface.raw.downcast_ref()
    }
}

#[cfg(feature = "gles")]
impl HalApi for hal::api::Gles {
    const VARIANT: Backend = Backend::Gl;
    fn create_instance_from_hal(name: &str, hal_instance: Self::Instance) -> Instance {
        #[allow(clippy::needless_update)]
        Instance {
            name: name.to_owned(),
            gl: Some(hal_instance),
            ..Default::default()
        }
    }
    fn instance_as_hal(instance: &Instance) -> Option<&Self::Instance> {
        instance.gl.as_ref()
    }
    fn hub<G: GlobalIdentityHandlerFactory>(global: &Global<G>) -> &Hub<Self> {
        &global.hubs.gl
    }
    fn get_surface(surface: &Surface) -> Option<&HalSurface<Self>> {
        surface.raw.downcast_ref()
    }
}
