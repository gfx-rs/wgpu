use wgt::{Backend, WasmNotSendSync};

use crate::{global::Global, hub::Hub};

pub trait HalApi: hal::Api + 'static + WasmNotSendSync {
    const VARIANT: Backend;

    fn hub(global: &Global) -> &Hub<Self>;
}

impl HalApi for hal::api::Empty {
    const VARIANT: Backend = Backend::Empty;

    fn hub(_: &Global) -> &Hub<Self> {
        unimplemented!("called empty api")
    }
}

#[cfg(vulkan)]
impl HalApi for hal::api::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;

    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.vulkan
    }
}

#[cfg(metal)]
impl HalApi for hal::api::Metal {
    const VARIANT: Backend = Backend::Metal;

    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.metal
    }
}

#[cfg(dx12)]
impl HalApi for hal::api::Dx12 {
    const VARIANT: Backend = Backend::Dx12;

    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.dx12
    }
}

#[cfg(gles)]
impl HalApi for hal::api::Gles {
    const VARIANT: Backend = Backend::Gl;

    fn hub(global: &Global) -> &Hub<Self> {
        &global.hubs.gl
    }
}
