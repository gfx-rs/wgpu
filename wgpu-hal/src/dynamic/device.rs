use crate::{Device, DynBuffer, DynResource};

use super::DynResourceExt;

pub trait DynDevice: DynResource {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);
}

impl<D: Device + DynResource> DynDevice for D {
    unsafe fn destroy_buffer(&self, mut buffer: Box<dyn DynBuffer>) {
        let buffer = buffer.expect_downcast_mut();
        unsafe { self.destroy_buffer(buffer) };
    }
}
