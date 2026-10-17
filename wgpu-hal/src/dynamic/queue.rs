use alloc::{boxed::Box, vec::Vec};

use crate::{
    DeviceError, DynCommandBuffer, DynFence, DynResource, DynSurface, DynSurfaceTexture,
    FenceValue, Queue, SurfaceError,
};

use super::DynResourceExt as _;

pub trait DynQueue: DynResource {
    /// Dynamic version of [Queue::submit]
    unsafe fn submit(
        &self,
        command_buffers: &[&dyn DynCommandBuffer],
        surface_textures: &[&dyn DynSurfaceTexture],
        signal_fence: (&dyn DynFence, FenceValue),
    ) -> Result<(), DeviceError>;
    /// Dynamic version of [Queue::present]
    unsafe fn present(
        &self,
        surface: &dyn DynSurface,
        texture: Box<dyn DynSurfaceTexture>,
    ) -> Result<(), SurfaceError>;
    /// Dynamic version of [Queue::get_timestamp_period]
    unsafe fn get_timestamp_period(&self) -> f32;
    /// Dynamic version of [Queue::wait_for_idle]
    unsafe fn wait_for_idle(&self) -> Result<(), DeviceError>;
}

impl<Q: Queue + DynResource> DynQueue for Q {
    unsafe fn submit(
        &self,
        command_buffers: &[&dyn DynCommandBuffer],
        surface_textures: &[&dyn DynSurfaceTexture],
        signal_fence: (&dyn DynFence, FenceValue),
    ) -> Result<(), DeviceError> {
        let command_buffers = command_buffers
            .iter()
            .map(|cb| (*cb).expect_downcast_ref())
            .collect::<Vec<_>>();
        let surface_textures = surface_textures
            .iter()
            .map(|surface| (*surface).expect_downcast_ref())
            .collect::<Vec<_>>();
        let signal_fence = (signal_fence.0.expect_downcast_ref(), signal_fence.1);
        unsafe { Q::submit(self, &command_buffers, &surface_textures, signal_fence) }
    }

    unsafe fn present(
        &self,
        surface: &dyn DynSurface,
        texture: Box<dyn DynSurfaceTexture>,
    ) -> Result<(), SurfaceError> {
        let surface = surface.expect_downcast_ref();
        unsafe { Q::present(self, surface, texture.unbox()) }
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        unsafe { Q::get_timestamp_period(self) }
    }

    unsafe fn wait_for_idle(&self) -> Result<(), DeviceError> {
        unsafe { Q::wait_for_idle(self) }
    }
}
