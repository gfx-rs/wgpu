use std::ops::Range;

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;
use crate::resource::{CBindGroup, CComputePipeline, CQuerySet, CRenderBundle, CRenderPipeline};

// ── CComputePass ──────────────────────────────────────────────────────────────

pub struct CComputePass {
    pub(crate) ptr: native::WGPUComputePassEncoder,
}

impl std::fmt::Debug for CComputePass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CComputePass")
            .field("ptr", &self.ptr)
            .finish()
    }
}

unsafe impl Send for CComputePass {}
unsafe impl Sync for CComputePass {}

impl Drop for CComputePass {
    fn drop(&mut self) {
        unsafe {
            wgpuComputePassEncoderEnd(self.ptr);
            wgpuComputePassEncoderRelease(self.ptr);
        }
    }
}

impl ComputePassInterface for CComputePass {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) {
        let ptr = pipeline.as_custom::<CComputePipeline>().unwrap().ptr;
        unsafe { wgpuComputePassEncoderSetPipeline(self.ptr, ptr) };
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let bg_ptr = bind_group
            .map(|bg| bg.as_custom::<CBindGroup>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());
        unsafe {
            wgpuComputePassEncoderSetBindGroup(
                self.ptr,
                index,
                bg_ptr,
                offsets.len(),
                offsets.as_ptr(),
            )
        };
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        unsafe {
            wgpuComputePassEncoderSetImmediates(self.ptr, offset, data.len() as u32, data.as_ptr())
        };
    }

    fn insert_debug_marker(&mut self, label: &str) {
        let sv = conv::str_to_string_view(label);
        unsafe { wgpuComputePassEncoderInsertDebugMarker(self.ptr, sv) };
    }

    fn push_debug_group(&mut self, group_label: &str) {
        let sv = conv::str_to_string_view(group_label);
        unsafe { wgpuComputePassEncoderPushDebugGroup(self.ptr, sv) };
    }

    fn pop_debug_group(&mut self) {
        unsafe { wgpuComputePassEncoderPopDebugGroup(self.ptr) };
    }

    fn write_timestamp(&mut self, query_set: &DispatchQuerySet, query_index: u32) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        unsafe { wgpuComputePassEncoderWriteTimestamp(self.ptr, qs_ptr, query_index) };
    }

    fn begin_pipeline_statistics_query(&mut self, query_set: &DispatchQuerySet, query_index: u32) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        unsafe {
            wgpuComputePassEncoderBeginPipelineStatisticsQuery(self.ptr, qs_ptr, query_index)
        };
    }

    fn end_pipeline_statistics_query(&mut self) {
        unsafe { wgpuComputePassEncoderEndPipelineStatisticsQuery(self.ptr) };
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        unsafe { wgpuComputePassEncoderDispatchWorkgroups(self.ptr, x, y, z) };
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(self.ptr, buf_ptr, indirect_offset)
        };
    }

    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>,
        >,
        _texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a DispatchTextureView>,
        >,
    ) {
        // wgpu-native has no explicit resource transition API.
        unimplemented!("wgpu-native does not expose explicit resource transitions")
    }
}

// ── CRenderPass ───────────────────────────────────────────────────────────────

pub struct CRenderPass {
    pub(crate) ptr: native::WGPURenderPassEncoder,
}

impl std::fmt::Debug for CRenderPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CRenderPass")
            .field("ptr", &self.ptr)
            .finish()
    }
}

unsafe impl Send for CRenderPass {}
unsafe impl Sync for CRenderPass {}

impl Drop for CRenderPass {
    fn drop(&mut self) {
        unsafe {
            wgpuRenderPassEncoderEnd(self.ptr);
            wgpuRenderPassEncoderRelease(self.ptr);
        }
    }
}

impl RenderPassInterface for CRenderPass {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) {
        let ptr = pipeline.as_custom::<CRenderPipeline>().unwrap().ptr;
        unsafe { wgpuRenderPassEncoderSetPipeline(self.ptr, ptr) };
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let bg_ptr = bind_group
            .map(|bg| bg.as_custom::<CBindGroup>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());
        unsafe {
            wgpuRenderPassEncoderSetBindGroup(
                self.ptr,
                index,
                bg_ptr,
                offsets.len(),
                offsets.as_ptr(),
            )
        };
    }

    fn set_index_buffer(
        &mut self,
        buffer: &DispatchBuffer,
        index_format: wgpu::IndexFormat,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        let buf_ptr = buffer.as_custom::<crate::resource::CBuffer>().unwrap().ptr;
        let c_format = conv::index_format_to_native(index_format);
        let c_size = size.map(|s| s.get()).unwrap_or(u64::MAX);
        unsafe { wgpuRenderPassEncoderSetIndexBuffer(self.ptr, buf_ptr, c_format, offset, c_size) };
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&DispatchBuffer>,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        let buf_ptr = buffer
            .map(|b| b.as_custom::<crate::resource::CBuffer>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());
        let c_size = size.map(|s| s.get()).unwrap_or(u64::MAX);
        unsafe { wgpuRenderPassEncoderSetVertexBuffer(self.ptr, slot, buf_ptr, offset, c_size) };
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        unsafe {
            wgpuRenderPassEncoderSetImmediates(self.ptr, offset, data.len() as u32, data.as_ptr())
        };
    }

    fn set_blend_constant(&mut self, color: wgpu::Color) {
        let c = conv::color_to_native(color);
        unsafe { wgpuRenderPassEncoderSetBlendConstant(self.ptr, Some(&c)) };
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        unsafe { wgpuRenderPassEncoderSetScissorRect(self.ptr, x, y, width, height) };
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        unsafe {
            wgpuRenderPassEncoderSetViewport(self.ptr, x, y, width, height, min_depth, max_depth)
        };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { wgpuRenderPassEncoderSetStencilReference(self.ptr, reference) };
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        unsafe {
            wgpuRenderPassEncoderDraw(
                self.ptr,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        };
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        unsafe {
            wgpuRenderPassEncoderDrawIndexed(
                self.ptr,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        };
    }

    fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            wgpuRenderPassEncoderDrawMeshTasks(
                self.ptr,
                group_count_x,
                group_count_y,
                group_count_z,
            )
        };
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe { wgpuRenderPassEncoderDrawIndirect(self.ptr, buf_ptr, indirect_offset) };
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe { wgpuRenderPassEncoderDrawIndexedIndirect(self.ptr, buf_ptr, indirect_offset) };
    }

    fn draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe { wgpuRenderPassEncoderDrawMeshTasksIndirect(self.ptr, buf_ptr, indirect_offset) };
    }

    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawIndirect(self.ptr, buf_ptr, indirect_offset, count)
        };
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawIndexedIndirect(self.ptr, buf_ptr, indirect_offset, count)
        };
    }

    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        let cnt_ptr = count_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawIndirectCount(
                self.ptr,
                buf_ptr,
                indirect_offset,
                cnt_ptr,
                count_buffer_offset,
                max_count,
            )
        };
    }

    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawMeshTasksIndirect(
                self.ptr,
                buf_ptr,
                indirect_offset,
                count,
            )
        };
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        let cnt_ptr = count_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(
                self.ptr,
                buf_ptr,
                indirect_offset,
                cnt_ptr,
                count_buffer_offset,
                max_count,
            )
        };
    }

    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        let buf_ptr = indirect_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        let cnt_ptr = count_buffer
            .as_custom::<crate::resource::CBuffer>()
            .unwrap()
            .ptr;
        unsafe {
            wgpuRenderPassEncoderMultiDrawMeshTasksIndirectCount(
                self.ptr,
                buf_ptr,
                indirect_offset,
                cnt_ptr,
                count_buffer_offset,
                max_count,
            )
        };
    }

    fn insert_debug_marker(&mut self, label: &str) {
        let sv = conv::str_to_string_view(label);
        unsafe { wgpuRenderPassEncoderInsertDebugMarker(self.ptr, sv) };
    }

    fn push_debug_group(&mut self, group_label: &str) {
        let sv = conv::str_to_string_view(group_label);
        unsafe { wgpuRenderPassEncoderPushDebugGroup(self.ptr, sv) };
    }

    fn pop_debug_group(&mut self) {
        unsafe { wgpuRenderPassEncoderPopDebugGroup(self.ptr) };
    }

    fn write_timestamp(&mut self, query_set: &DispatchQuerySet, query_index: u32) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        unsafe { wgpuRenderPassEncoderWriteTimestamp(self.ptr, qs_ptr, query_index) };
    }

    fn begin_occlusion_query(&mut self, query_index: u32) {
        unsafe { wgpuRenderPassEncoderBeginOcclusionQuery(self.ptr, query_index) };
    }

    fn end_occlusion_query(&mut self) {
        unsafe { wgpuRenderPassEncoderEndOcclusionQuery(self.ptr) };
    }

    fn begin_pipeline_statistics_query(&mut self, query_set: &DispatchQuerySet, query_index: u32) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        unsafe { wgpuRenderPassEncoderBeginPipelineStatisticsQuery(self.ptr, qs_ptr, query_index) };
    }

    fn end_pipeline_statistics_query(&mut self) {
        unsafe { wgpuRenderPassEncoderEndPipelineStatisticsQuery(self.ptr) };
    }

    fn execute_bundles(&mut self, render_bundles: &mut dyn Iterator<Item = &DispatchRenderBundle>) {
        let ptrs: Vec<native::WGPURenderBundle> = render_bundles
            .map(|b| b.as_custom::<CRenderBundle>().unwrap().ptr)
            .collect();
        unsafe { wgpuRenderPassEncoderExecuteBundles(self.ptr, ptrs.len(), ptrs.as_ptr()) };
    }
}
