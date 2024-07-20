use crate::{Device, DynBuffer};

use super::DynResourceExt;

pub trait DynDevice {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);
}

impl<D: Device> DynDevice for D {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>) {
        unsafe { D::destroy_buffer(self, buffer.unbox()) };
    }
}
