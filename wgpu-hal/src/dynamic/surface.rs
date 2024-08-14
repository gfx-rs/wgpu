use crate::{
    DynDevice, DynFence, DynResource, DynSurfaceTexture, Surface, SurfaceConfiguration,
    SurfaceError,
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
    ) -> Result<(), SurfaceError>;

    unsafe fn unconfigure(&self, device: &dyn DynDevice);

    unsafe fn acquire_texture(
        &self,
        timeout: Option<std::time::Duration>,
        fence: &dyn DynFence,
    ) -> Result<Option<DynAcquiredSurfaceTexture>, SurfaceError>;

    unsafe fn discard_texture(&self, texture: Box<dyn DynSurfaceTexture>);
}

impl<S: Surface + DynResource> DynSurface for S {
    unsafe fn configure(
        &self,
        device: &dyn DynDevice,
        config: &SurfaceConfiguration,
    ) -> Result<(), SurfaceError> {
        let device = device.expect_downcast_ref();
        unsafe { S::configure(self, device, config) }
    }

    unsafe fn unconfigure(&self, device: &dyn DynDevice) {
        let device = device.expect_downcast_ref();
        unsafe { S::unconfigure(self, device) }
    }

    unsafe fn acquire_texture(
        &self,
        timeout: Option<std::time::Duration>,
        fence: &dyn DynFence,
    ) -> Result<Option<DynAcquiredSurfaceTexture>, SurfaceError> {
        let fence = fence.expect_downcast_ref();
        unsafe { S::acquire_texture(self, timeout, fence) }.map(|acquired| {
            acquired.map(|ast| {
                let texture = Box::new(ast.texture);
                let suboptimal = ast.suboptimal;
                DynAcquiredSurfaceTexture {
                    texture,
                    suboptimal,
                }
            })
        })
    }

    unsafe fn discard_texture(&self, texture: Box<dyn DynSurfaceTexture>) {
        unsafe { S::discard_texture(self, texture.unbox()) }
    }
}
