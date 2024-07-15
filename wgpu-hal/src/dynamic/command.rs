use crate::{
    BufferBarrier, BufferBinding, BufferCopy, CommandEncoder, DeviceError, Label, MemoryRange,
};

use super::{DynBuffer, DynResourceExt as _};

pub trait DynCommandEncoder: std::fmt::Debug {
    unsafe fn begin_encoding(&mut self, label: Label) -> Result<(), DeviceError>;

    unsafe fn discard_encoding(&mut self);

    unsafe fn transition_buffers(&mut self, barriers: &[BufferBarrier<'_, dyn DynBuffer>]);

    unsafe fn clear_buffer(&mut self, buffer: &dyn DynBuffer, range: MemoryRange);

    unsafe fn copy_buffer_to_buffer(
        &mut self,
        src: &dyn DynBuffer,
        dst: &dyn DynBuffer,
        regions: &[BufferCopy],
    );

    unsafe fn insert_debug_marker(&mut self, label: &str);
    unsafe fn begin_debug_marker(&mut self, group_label: &str);
    unsafe fn end_debug_marker(&mut self);

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, dyn DynBuffer>,
        format: wgt::IndexFormat,
    );

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: BufferBinding<'a, dyn DynBuffer>,
    );
}

impl<C: CommandEncoder> DynCommandEncoder for C {
    unsafe fn begin_encoding(&mut self, label: Label) -> Result<(), DeviceError> {
        unsafe { C::begin_encoding(self, label) }
    }

    unsafe fn discard_encoding(&mut self) {
        unsafe { C::discard_encoding(self) }
    }

    unsafe fn transition_buffers(&mut self, barriers: &[BufferBarrier<'_, dyn DynBuffer>]) {
        let barriers = barriers.iter().map(|barrier| BufferBarrier {
            buffer: barrier.buffer.expect_downcast_ref(),
            usage: barrier.usage.clone(),
        });
        unsafe { self.transition_buffers(barriers) };
    }

    unsafe fn clear_buffer(&mut self, buffer: &dyn DynBuffer, range: MemoryRange) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { C::clear_buffer(self, buffer, range) };
    }

    unsafe fn copy_buffer_to_buffer(
        &mut self,
        src: &dyn DynBuffer,
        dst: &dyn DynBuffer,
        regions: &[BufferCopy],
    ) {
        let src = src.expect_downcast_ref();
        let dst = dst.expect_downcast_ref();
        unsafe {
            C::copy_buffer_to_buffer(self, src, dst, regions.iter().copied());
        }
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        unsafe {
            C::insert_debug_marker(self, label);
        }
    }

    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        unsafe {
            C::begin_debug_marker(self, group_label);
        }
    }

    unsafe fn end_debug_marker(&mut self) {
        unsafe {
            C::end_debug_marker(self);
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, dyn DynBuffer>,
        format: wgt::IndexFormat,
    ) {
        let binding = binding.expect_downcast();
        unsafe { self.set_index_buffer(binding, format) };
    }

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: BufferBinding<'a, dyn DynBuffer>,
    ) {
        let binding = binding.expect_downcast();
        unsafe { self.set_vertex_buffer(index, binding) };
    }
}
