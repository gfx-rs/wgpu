use alloc::boxed::Box;
use core::time::Duration;

use crate::{
    DynDevice, DynFence, DynResource, DynSurfaceTexture, RawSurfaceConfiguration, Surface, SurfaceConfiguration, SurfaceError
};

use super::DynResourceExt as _;

#[derive(Debug)]
pub struct DynAcquiredSurfaceTexture {
    pub texture: Box<dyn DynSurfaceTexture>,
    /// The presentation configuration no longer matches
    /// the surface properties exactly, but can still be used to present
    /// to the surface successfully.
    pub suboptimal: bool,
}

pub trait DynSurface: DynResource {
    unsafe fn configure(
        &self,
        device: &dyn DynDevice,
        config: &SurfaceConfiguration,
        raw_config: Option<Box<dyn RawSurfaceConfiguration>>,
    ) -> Result<(), SurfaceError>;

    unsafe fn unconfigure(&self, device: &dyn DynDevice);

    unsafe fn acquire_texture(
        &self,
        timeout: Option<Duration>,
        fence: &dyn DynFence,
    ) -> Result<DynAcquiredSurfaceTexture, SurfaceError>;

    unsafe fn discard_texture(&self, texture: Box<dyn DynSurfaceTexture>);
}

impl<S: Surface + DynResource> DynSurface for S {
    unsafe fn configure(
        &self,
        device: &dyn DynDevice,
        config: &SurfaceConfiguration,
        raw_config: Option<Box<dyn RawSurfaceConfiguration>>,
    ) -> Result<(), SurfaceError> {
        // Normally, we would downcast `config.raw` here as well, but surfaces are special
        // because Vulkan can have either a native or DXGI swapchain, so we defer
        // downcasting to the backend.
        let device = device.expect_downcast_ref();
        unsafe { S::configure(self, device, config, raw_config) }
    }

    unsafe fn unconfigure(&self, device: &dyn DynDevice) {
        let device = device.expect_downcast_ref();
        unsafe { S::unconfigure(self, device) }
    }

    unsafe fn acquire_texture(
        &self,
        timeout: Option<Duration>,
        fence: &dyn DynFence,
    ) -> Result<DynAcquiredSurfaceTexture, SurfaceError> {
        let fence = fence.expect_downcast_ref();
        unsafe { S::acquire_texture(self, timeout, fence) }.map(|ast| {
            let texture = Box::new(ast.texture);
            let suboptimal = ast.suboptimal;
            DynAcquiredSurfaceTexture {
                texture,
                suboptimal,
            }
        })
    }

    unsafe fn discard_texture(&self, texture: Box<dyn DynSurfaceTexture>) {
        unsafe { S::discard_texture(self, texture.unbox()) }
    }
}
