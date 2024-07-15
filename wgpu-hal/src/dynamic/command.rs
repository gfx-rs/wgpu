use std::ops::Range;

use crate::{
    BufferBarrier, BufferBinding, BufferCopy, CommandEncoder, DeviceError, DynBindGroup,
    DynPipelineLayout, DynQuerySet, Label, MemoryRange,
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

    unsafe fn set_bind_group(
        &mut self,
        layout: &dyn DynPipelineLayout,
        index: u32,
        group: &dyn DynBindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    );

    unsafe fn set_push_constants(
        &mut self,
        layout: &dyn DynPipelineLayout,
        stages: wgt::ShaderStages,
        offset_bytes: u32,
        data: &[u32],
    );

    unsafe fn insert_debug_marker(&mut self, label: &str);
    unsafe fn begin_debug_marker(&mut self, group_label: &str);
    unsafe fn end_debug_marker(&mut self);

    unsafe fn begin_query(&mut self, set: &dyn DynQuerySet, index: u32);
    unsafe fn end_query(&mut self, set: &dyn DynQuerySet, index: u32);
    unsafe fn write_timestamp(&mut self, set: &dyn DynQuerySet, index: u32);
    unsafe fn reset_queries(&mut self, set: &dyn DynQuerySet, range: Range<u32>);
    unsafe fn copy_query_results(
        &mut self,
        set: &dyn DynQuerySet,
        range: Range<u32>,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    );

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

    unsafe fn set_bind_group(
        &mut self,
        layout: &dyn DynPipelineLayout,
        index: u32,
        group: &dyn DynBindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let layout = layout.expect_downcast_ref();
        let group = group.expect_downcast_ref();
        unsafe { C::set_bind_group(self, layout, index, group, dynamic_offsets) };
    }

    unsafe fn set_push_constants(
        &mut self,
        layout: &dyn DynPipelineLayout,
        stages: wgt::ShaderStages,
        offset_bytes: u32,
        data: &[u32],
    ) {
        let layout = layout.expect_downcast_ref();
        unsafe { C::set_push_constants(self, layout, stages, offset_bytes, data) };
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

    unsafe fn begin_query(&mut self, set: &dyn DynQuerySet, index: u32) {
        let set = set.expect_downcast_ref();
        unsafe { C::begin_query(self, set, index) };
    }

    unsafe fn end_query(&mut self, set: &dyn DynQuerySet, index: u32) {
        let set = set.expect_downcast_ref();
        unsafe { C::end_query(self, set, index) };
    }

    unsafe fn write_timestamp(&mut self, set: &dyn DynQuerySet, index: u32) {
        let set = set.expect_downcast_ref();
        unsafe { C::write_timestamp(self, set, index) };
    }

    unsafe fn reset_queries(&mut self, set: &dyn DynQuerySet, range: Range<u32>) {
        let set = set.expect_downcast_ref();
        unsafe { C::reset_queries(self, set, range) };
    }

    unsafe fn copy_query_results(
        &mut self,
        set: &dyn DynQuerySet,
        range: Range<u32>,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
        let set = set.expect_downcast_ref();
        let buffer = buffer.expect_downcast_ref();
        unsafe { C::copy_query_results(self, set, range, buffer, offset, stride) };
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
