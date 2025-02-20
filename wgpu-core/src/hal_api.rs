use wgt::{Backend, WasmNotSendSync};

pub trait HalApi: hal::Api + 'static + WasmNotSendSync {
    const VARIANT: Backend;
}

impl HalApi for hal::api::Noop {
    const VARIANT: Backend = Backend::Noop;
}

#[cfg(vulkan)]
impl HalApi for hal::api::Vulkan {
    const VARIANT: Backend = Backend::Vulkan;
}

#[cfg(metal)]
impl HalApi for hal::api::Metal {
    const VARIANT: Backend = Backend::Metal;
}

#[cfg(dx12)]
impl HalApi for hal::api::Dx12 {
    const VARIANT: Backend = Backend::Dx12;
}

#[cfg(gles)]
impl HalApi for hal::api::Gles {
    const VARIANT: Backend = Backend::Gl;
}
