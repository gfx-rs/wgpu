use alloc::{boxed::Box, vec::Vec};

use crate::{
    Adapter, Api, DeviceError, OpenDevice, SurfaceCapabilities, TextureFormatCapabilities,
};

use super::{DynDevice, DynQueue, DynResource, DynResourceExt, DynSurface};

pub struct DynOpenDevice {
    pub device: Box<dyn DynDevice>,
    pub queues: Vec<Box<dyn DynQueue>>,
}

impl<A: Api> From<OpenDevice<A>> for DynOpenDevice {
    fn from(open_device: OpenDevice<A>) -> Self {
        Self {
            device: Box::new(open_device.device),
            queues: open_device
                .queues
                .into_iter()
                .map(|q| {
                    let b: Box<dyn DynQueue> = Box::new(q);
                    b
                })
                .collect(),
        }
    }
}

pub trait DynAdapter: DynResource {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
        queues: &[u32],
    ) -> Result<DynOpenDevice, DeviceError>;

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapabilities;

    unsafe fn surface_capabilities(&self, surface: &dyn DynSurface) -> Option<SurfaceCapabilities>;

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp;
}

impl<A: Adapter + DynResource> DynAdapter for A {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
        queues: &[u32],
    ) -> Result<DynOpenDevice, DeviceError> {
        unsafe { A::open(self, features, limits, memory_hints, queues) }.map(|open_device| {
            DynOpenDevice {
                device: Box::new(open_device.device),
                queues: open_device
                    .queues
                    .into_iter()
                    .map(|q| {
                        let b: Box<dyn DynQueue> = Box::new(q);
                        b
                    })
                    .collect(),
            }
        })
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapabilities {
        unsafe { A::texture_format_capabilities(self, format) }
    }

    unsafe fn surface_capabilities(&self, surface: &dyn DynSurface) -> Option<SurfaceCapabilities> {
        let surface = surface.expect_downcast_ref();
        unsafe { A::surface_capabilities(self, surface) }
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        unsafe { A::get_presentation_timestamp(self) }
    }
}
