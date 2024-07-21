use crate::{Device, DynBuffer, DynResource};

use super::DynResourceExt;

pub trait DynDevice: DynResource {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);
}

impl<D: Device + DynResource> DynDevice for D {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>) {
        unsafe { D::destroy_buffer(self, buffer.unbox()) };
    }
}
