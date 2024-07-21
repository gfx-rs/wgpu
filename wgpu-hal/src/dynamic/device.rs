use crate::{
    BufferDescriptor, BufferMapping, Device, DeviceError, DynBuffer, DynResource, MemoryRange,
};

use super::DynResourceExt as _;

pub trait DynDevice: DynResource {
    unsafe fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<Box<dyn DynBuffer>, DeviceError>;

    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);

    unsafe fn map_buffer(
        &self,
        buffer: &dyn DynBuffer,
        range: MemoryRange,
    ) -> Result<BufferMapping, DeviceError>;

    unsafe fn unmap_buffer(&self, buffer: &dyn DynBuffer);

    unsafe fn flush_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]);
    unsafe fn invalidate_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]);
}

impl<D: Device + DynResource> DynDevice for D {
    unsafe fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<Box<dyn DynBuffer>, DeviceError> {
        unsafe { D::create_buffer(self, desc) }.map(|b| -> Box<dyn DynBuffer> { Box::new(b) })
    }

    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>) {
        unsafe { D::destroy_buffer(self, buffer.unbox()) };
    }

    unsafe fn map_buffer(
        &self,
        buffer: &dyn DynBuffer,
        range: MemoryRange,
    ) -> Result<BufferMapping, DeviceError> {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::map_buffer(self, buffer, range) }
    }

    unsafe fn unmap_buffer(&self, buffer: &dyn DynBuffer) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::unmap_buffer(self, buffer) }
    }

    unsafe fn flush_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::flush_mapped_ranges(self, buffer, ranges.iter().cloned()) }
    }

    unsafe fn invalidate_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::invalidate_mapped_ranges(self, buffer, ranges.iter().cloned()) }
    }
}
