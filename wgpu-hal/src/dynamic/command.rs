use std::ops::Range;

use crate::{
    BufferBarrier, BufferBinding, BufferCopy, CommandEncoder, DeviceError, DynBuffer, Label,
    MemoryRange, Rect,
};

use super::DynResourceExt as _;

pub trait DynCommandEncoder: std::fmt::Debug {
    unsafe fn begin_encoding(&mut self, label: Label) -> Result<(), DeviceError>;

    unsafe fn discard_encoding(&mut self);

    //unsafe fn end_encoding(&mut self) -> Result<dyn DynCommandBuffer, DeviceError>;

    // unsafe fn reset_all<I>(&mut self, command_buffers: I)
    // where
    //     I: Iterator<Item = dyn DynCommandBuffer>;

    unsafe fn transition_buffers(&mut self, barriers: &[BufferBarrier<'_, dyn DynBuffer>]);

    // unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    // where
    //     T: Iterator<Item = TextureBarrier<'a, Self::A>>;

    unsafe fn clear_buffer(&mut self, buffer: &dyn DynBuffer, range: MemoryRange);

    unsafe fn copy_buffer_to_buffer(
        &mut self,
        src: &dyn DynBuffer,
        dst: &dyn DynBuffer,
        regions: &[BufferCopy],
    );

    // #[cfg(webgl)]
    // unsafe fn copy_external_image_to_texture<T>(
    //     &mut self,
    //     src: &wgt::ImageCopyExternalImage,
    //     dst: &dyn DynTexture,
    //     dst_premultiplication: bool,
    //     regions: T,
    // ) where
    //     T: Iterator<Item = TextureCopy>;

    // unsafe fn copy_texture_to_texture<T>(
    //     &mut self,
    //     src: &dyn DynTexture,
    //     src_usage: TextureUses,
    //     dst: &dyn DynTexture,
    //     regions: T,
    // ) where
    //     T: Iterator<Item = TextureCopy>;

    // unsafe fn copy_buffer_to_texture<T>(
    //     &mut self,
    //     src: &dyn DynBuffer,
    //     dst: &dyn DynTexture,
    //     regions: T,
    // ) where
    //     T: Iterator<Item = BufferTextureCopy>;

    // unsafe fn copy_texture_to_buffer<T>(
    //     &mut self,
    //     src: &dyn DynTexture,
    //     src_usage: TextureUses,
    //     dst: &dyn DynBuffer,
    //     regions: T,
    // ) where
    //     T: Iterator<Item = BufferTextureCopy>;

    // unsafe fn set_bind_group(
    //     &mut self,
    //     layout: &dyn DynPipelineLayout,
    //     index: u32,
    //     group: &dyn DynBindGroup,
    //     dynamic_offsets: &[wgt::DynamicOffset],
    // );

    // unsafe fn set_push_constants(
    //     &mut self,
    //     layout: &dyn DynPipelineLayout,
    //     stages: wgt::ShaderStages,
    //     offset_bytes: u32,
    //     data: &[u32],
    // );

    unsafe fn insert_debug_marker(&mut self, label: &str);
    unsafe fn begin_debug_marker(&mut self, group_label: &str);
    unsafe fn end_debug_marker(&mut self);

    // unsafe fn begin_query(&mut self, set: &dyn DynQuerySet, index: u32);
    // unsafe fn end_query(&mut self, set: &dyn DynQuerySet, index: u32);
    // unsafe fn write_timestamp(&mut self, set: &dyn DynQuerySet, index: u32);
    // unsafe fn reset_queries(&mut self, set: &dyn DynQuerySet, range: Range<u32>);
    // unsafe fn copy_query_results(
    //     &mut self,
    //     set: &dyn DynQuerySet,
    //     range: Range<u32>,
    //     buffer: &dyn DynBuffer,
    //     offset: wgt::BufferAddress,
    //     stride: wgt::BufferSize,
    // );

    // unsafe fn begin_render_pass(&mut self, desc: &RenderPassDescriptor<Self::A>);
    // unsafe fn end_render_pass(&mut self);

    //unsafe fn set_render_pipeline(&mut self, pipeline: &dyn DynRenderPipeline);

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
    unsafe fn set_viewport(&mut self, rect: &Rect<f32>, depth_range: Range<f32>);
    unsafe fn set_scissor_rect(&mut self, rect: &Rect<u32>);
    unsafe fn set_stencil_reference(&mut self, value: u32);
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]);

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    );
    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    );
    unsafe fn draw_indirect(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    );
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    );
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    );
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    );

    // unsafe fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor<Self::A>);
    // unsafe fn end_compute_pass(&mut self);

    //unsafe fn set_compute_pipeline(&mut self, pipeline: &dyn DynComputePipeline);

    unsafe fn dispatch(&mut self, count: [u32; 3]);
    unsafe fn dispatch_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress);

    // unsafe fn build_acceleration_structures<'a, T>(
    //     &mut self,
    //     descriptor_count: u32,
    //     descriptors: T,
    // ) where
    //     Self::A: 'a,
    //     T: IntoIterator<Item = BuildAccelerationStructureDescriptor<'a, Self::A>>;

    // unsafe fn place_acceleration_structure_barrier(
    //     &mut self,
    //     barrier: AccelerationStructureBarrier,
    // );
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

    unsafe fn set_viewport(&mut self, rect: &Rect<f32>, depth_range: Range<f32>) {
        unsafe {
            C::set_viewport(self, rect, depth_range);
        }
    }

    unsafe fn set_scissor_rect(&mut self, rect: &Rect<u32>) {
        unsafe {
            C::set_scissor_rect(self, rect);
        }
    }

    unsafe fn set_stencil_reference(&mut self, value: u32) {
        unsafe {
            C::set_stencil_reference(self, value);
        }
    }

    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        unsafe { C::set_blend_constants(self, color) };
    }

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            C::draw(
                self,
                first_vertex,
                vertex_count,
                first_instance,
                instance_count,
            )
        };
    }

    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            C::draw_indexed(
                self,
                first_index,
                index_count,
                base_vertex,
                first_instance,
                instance_count,
            )
        };
    }

    unsafe fn draw_indirect(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { C::draw_indirect(self, buffer, offset, draw_count) };
    }

    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { C::draw_indexed_indirect(self, buffer, offset, draw_count) };
    }

    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        let buffer = buffer.expect_downcast_ref();
        let count_buffer = count_buffer.expect_downcast_ref();
        unsafe {
            C::draw_indirect_count(self, buffer, offset, count_buffer, count_offset, max_count)
        };
    }

    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        let buffer = buffer.expect_downcast_ref();
        let count_buffer = count_buffer.expect_downcast_ref();
        unsafe {
            C::draw_indexed_indirect_count(
                self,
                buffer,
                offset,
                count_buffer,
                count_offset,
                max_count,
            )
        };
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        unsafe { C::dispatch(self, count) };
    }

    unsafe fn dispatch_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { C::dispatch_indirect(self, buffer, offset) };
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, dyn DynBuffer>,
        format: wgt::IndexFormat,
    ) {
        let binding = binding.expect_downcast();
        unsafe { C::set_index_buffer(self, binding, format) };
    }

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: BufferBinding<'a, dyn DynBuffer>,
    ) {
        let binding = binding.expect_downcast();
        unsafe { C::set_vertex_buffer(self, index, binding) };
    }
}
