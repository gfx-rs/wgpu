use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;
use crate::pass::{CComputePass, CRenderPass};
use crate::resource::*;

// ── CCommandEncoder ───────────────────────────────────────────────────────────

pub struct CCommandEncoder {
    pub(crate) ptr: native::WGPUCommandEncoder,
    pub(crate) device_ptr: native::WGPUDevice,
}
impl std::fmt::Debug for CCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CCommandEncoder")
            .field("ptr", &self.ptr)
            .finish()
    }
}
unsafe impl Send for CCommandEncoder {}
unsafe impl Sync for CCommandEncoder {}

impl Drop for CCommandEncoder {
    fn drop(&mut self) {
        unsafe { wgpuCommandEncoderRelease(self.ptr) };
    }
}

impl CommandEncoderInterface for CCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &DispatchBuffer,
        source_offset: wgpu::BufferAddress,
        destination: &DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
        copy_size: Option<wgpu::BufferAddress>,
    ) {
        let src_ptr = source.as_custom::<CBuffer>().unwrap().ptr;
        let dst_ptr = destination.as_custom::<CBuffer>().unwrap().ptr;
        // None means "copy to end": use remaining size. We pass WGPU_WHOLE_SIZE sentinel for None.
        let size = copy_size.unwrap_or(u64::MAX);
        unsafe {
            wgpuCommandEncoderCopyBufferToBuffer(
                self.ptr,
                src_ptr,
                source_offset,
                dst_ptr,
                destination_offset,
                size,
            )
        };
    }

    fn copy_buffer_to_texture(
        &self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src_ptr = source.buffer.as_custom::<CBuffer>().unwrap().ptr;
        let dst_ptr = destination.texture.as_custom::<CTexture>().unwrap().ptr;
        let c_src = conv::image_copy_buffer_to_native(&source, src_ptr);
        let c_dst = conv::image_copy_texture_to_native(&destination, dst_ptr);
        let c_size = conv::extent3d_to_native(copy_size);
        unsafe {
            wgpuCommandEncoderCopyBufferToTexture(
                self.ptr,
                Some(&c_src),
                Some(&c_dst),
                Some(&c_size),
            )
        };
    }

    fn copy_texture_to_buffer(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src_ptr = source.texture.as_custom::<CTexture>().unwrap().ptr;
        let dst_ptr = destination.buffer.as_custom::<CBuffer>().unwrap().ptr;
        let c_src = conv::image_copy_texture_to_native(&source, src_ptr);
        let c_dst = conv::image_copy_buffer_to_native(&destination, dst_ptr);
        let c_size = conv::extent3d_to_native(copy_size);
        unsafe {
            wgpuCommandEncoderCopyTextureToBuffer(
                self.ptr,
                Some(&c_src),
                Some(&c_dst),
                Some(&c_size),
            )
        };
    }

    fn copy_texture_to_texture(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src_ptr = source.texture.as_custom::<CTexture>().unwrap().ptr;
        let dst_ptr = destination.texture.as_custom::<CTexture>().unwrap().ptr;
        let c_src = conv::image_copy_texture_to_native(&source, src_ptr);
        let c_dst = conv::image_copy_texture_to_native(&destination, dst_ptr);
        let c_size = conv::extent3d_to_native(copy_size);
        unsafe {
            wgpuCommandEncoderCopyTextureToTexture(
                self.ptr,
                Some(&c_src),
                Some(&c_dst),
                Some(&c_size),
            )
        };
    }

    fn begin_compute_pass(&self, desc: &wgpu::ComputePassDescriptor<'_>) -> DispatchComputePass {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        // Convert optional timestamp writes.
        let ts_writes = desc.timestamp_writes.as_ref().map(|tw| {
            let qs_ptr = tw.query_set.as_custom::<CQuerySet>().unwrap().ptr;
            native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: qs_ptr,
                beginningOfPassWriteIndex: tw
                    .beginning_of_pass_write_index
                    .unwrap_or(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
                endOfPassWriteIndex: tw
                    .end_of_pass_write_index
                    .unwrap_or(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
            }
        });
        let ts_ptr: *const native::WGPUPassTimestampWrites = ts_writes
            .as_ref()
            .map(std::ptr::from_ref)
            .unwrap_or(std::ptr::null());

        let c_desc = native::WGPUComputePassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            timestampWrites: ts_ptr,
        };
        let ptr = unsafe { wgpuCommandEncoderBeginComputePass(self.ptr, Some(&c_desc)) };
        DispatchComputePass::custom(CComputePass { ptr })
    }

    fn begin_render_pass(&self, desc: &wgpu::RenderPassDescriptor<'_>) -> DispatchRenderPass {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        // Color attachments.
        let color_attachments: Vec<native::WGPURenderPassColorAttachment> = desc
            .color_attachments
            .iter()
            .map(|opt_att| {
                if let Some(att) = opt_att {
                    let view_ptr = att.view.as_custom::<CTextureView>().unwrap().ptr;
                    let resolve_ptr = att
                        .resolve_target
                        .map(|rv| rv.as_custom::<CTextureView>().unwrap().ptr)
                        .unwrap_or(std::ptr::null());
                    let (load_op, clear_value) = conv::load_op_color_to_native(&att.ops.load);
                    native::WGPURenderPassColorAttachment {
                        nextInChain: std::ptr::null_mut(),
                        view: view_ptr,
                        depthSlice: att
                            .depth_slice
                            .unwrap_or(native::WGPU_DEPTH_SLICE_UNDEFINED),
                        resolveTarget: resolve_ptr,
                        loadOp: load_op,
                        storeOp: conv::store_op_to_native(att.ops.store),
                        clearValue: clear_value,
                    }
                } else {
                    // Hole in color attachments array.
                    native::WGPURenderPassColorAttachment {
                        nextInChain: std::ptr::null_mut(),
                        view: std::ptr::null(),
                        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
                        resolveTarget: std::ptr::null(),
                        loadOp: native::WGPULoadOp_Undefined,
                        storeOp: native::WGPUStoreOp_Undefined,
                        clearValue: native::WGPUColor {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        },
                    }
                }
            })
            .collect();

        // Depth stencil attachment.
        let ds_attach = desc.depth_stencil_attachment.as_ref().map(|ds| {
            let view_ptr = ds.view.as_custom::<CTextureView>().unwrap().ptr;
            let (depth_load_op, depth_clear) = ds
                .depth_ops
                .as_ref()
                .map(conv::load_op_f32_to_native)
                .unwrap_or((native::WGPULoadOp_Undefined, f32::NAN));
            let depth_store_op = ds
                .depth_ops
                .as_ref()
                .map(|ops| conv::store_op_to_native(ops.store))
                .unwrap_or(native::WGPUStoreOp_Undefined);
            let (stencil_load_op, stencil_clear) = ds
                .stencil_ops
                .as_ref()
                .map(conv::load_op_u32_to_native)
                .unwrap_or((native::WGPULoadOp_Undefined, 0));
            let stencil_store_op = ds
                .stencil_ops
                .as_ref()
                .map(|ops| conv::store_op_to_native(ops.store))
                .unwrap_or(native::WGPUStoreOp_Undefined);
            native::WGPURenderPassDepthStencilAttachment {
                nextInChain: std::ptr::null_mut(),
                view: view_ptr,
                depthLoadOp: depth_load_op,
                depthStoreOp: depth_store_op,
                depthClearValue: depth_clear,
                depthReadOnly: (ds.depth_ops.is_none()) as native::WGPUBool,
                stencilLoadOp: stencil_load_op,
                stencilStoreOp: stencil_store_op,
                stencilClearValue: stencil_clear,
                stencilReadOnly: (ds.stencil_ops.is_none()) as native::WGPUBool,
            }
        });
        let ds_ptr: *const native::WGPURenderPassDepthStencilAttachment = ds_attach
            .as_ref()
            .map(std::ptr::from_ref)
            .unwrap_or(std::ptr::null());

        // Timestamp writes.
        let ts_writes = desc.timestamp_writes.as_ref().map(|tw| {
            let qs_ptr = tw.query_set.as_custom::<CQuerySet>().unwrap().ptr;
            native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: qs_ptr,
                beginningOfPassWriteIndex: tw
                    .beginning_of_pass_write_index
                    .unwrap_or(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
                endOfPassWriteIndex: tw
                    .end_of_pass_write_index
                    .unwrap_or(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
            }
        });
        let ts_ptr: *const native::WGPUPassTimestampWrites = ts_writes
            .as_ref()
            .map(std::ptr::from_ref)
            .unwrap_or(std::ptr::null());

        // Occlusion query set.
        let occlusion_qs = desc
            .occlusion_query_set
            .map(|qs| qs.as_custom::<CQuerySet>().unwrap().ptr)
            .unwrap_or(std::ptr::null());

        let c_desc = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            colorAttachmentCount: color_attachments.len(),
            colorAttachments: if color_attachments.is_empty() {
                std::ptr::null()
            } else {
                color_attachments.as_ptr()
            },
            depthStencilAttachment: ds_ptr,
            occlusionQuerySet: occlusion_qs,
            timestampWrites: ts_ptr,
        };

        let ptr = unsafe { wgpuCommandEncoderBeginRenderPass(self.ptr, Some(&c_desc)) };
        DispatchRenderPass::custom(CRenderPass { ptr })
    }

    fn finish(&mut self) -> DispatchCommandBuffer {
        let ptr = unsafe {
            wgpuCommandEncoderFinish(
                self.ptr,
                Some(&native::WGPUCommandBufferDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: conv::null_string_view(),
                }),
            )
        };
        DispatchCommandBuffer::custom(CCommandBuffer { ptr, device_ptr: self.device_ptr })
    }

    fn clear_texture(
        &self,
        texture: &DispatchTexture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        let tex_ptr = texture.as_custom::<CTexture>().unwrap().ptr;
        let c_range = native::WGPUImageSubresourceRange {
            aspect: conv::texture_aspect_to_native(subresource_range.aspect),
            baseMipLevel: subresource_range.base_mip_level,
            mipLevelCount: subresource_range
                .mip_level_count
                .unwrap_or(native::WGPU_MIP_LEVEL_COUNT_UNDEFINED),
            baseArrayLayer: subresource_range.base_array_layer,
            arrayLayerCount: subresource_range
                .array_layer_count
                .unwrap_or(native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED),
        };
        unsafe { wgpuCommandEncoderClearTexture(self.ptr, tex_ptr, Some(&c_range)) };
    }

    fn clear_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferAddress>,
    ) {
        let buf_ptr = buffer.as_custom::<CBuffer>().unwrap().ptr;
        let c_size = size.unwrap_or(u64::MAX);
        unsafe { wgpuCommandEncoderClearBuffer(self.ptr, buf_ptr, offset, c_size) };
    }

    fn insert_debug_marker(&self, label: &str) {
        let sv = conv::str_to_string_view(label);
        unsafe { wgpuCommandEncoderInsertDebugMarker(self.ptr, sv) };
    }

    fn push_debug_group(&self, label: &str) {
        let sv = conv::str_to_string_view(label);
        unsafe { wgpuCommandEncoderPushDebugGroup(self.ptr, sv) };
    }

    fn pop_debug_group(&self) {
        unsafe { wgpuCommandEncoderPopDebugGroup(self.ptr) };
    }

    fn write_timestamp(&self, query_set: &DispatchQuerySet, query_index: u32) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        unsafe { wgpuCommandEncoderWriteTimestamp(self.ptr, qs_ptr, query_index) };
    }

    fn resolve_query_set(
        &self,
        query_set: &DispatchQuerySet,
        first_query: u32,
        query_count: u32,
        destination: &DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
    ) {
        let qs_ptr = query_set.as_custom::<CQuerySet>().unwrap().ptr;
        let dst_ptr = destination.as_custom::<CBuffer>().unwrap().ptr;
        unsafe {
            wgpuCommandEncoderResolveQuerySet(
                self.ptr,
                qs_ptr,
                first_query,
                query_count,
                dst_ptr,
                destination_offset,
            )
        };
    }

    fn mark_acceleration_structures_built<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        // wgpu-native does not expose ray tracing acceleration structures.
        unimplemented!("wgpu-native does not expose acceleration structures")
    }

    fn build_acceleration_structures<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        // wgpu-native does not expose ray tracing acceleration structures.
        unimplemented!("wgpu-native does not expose acceleration structures")
    }

    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>,
        >,
        _texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a DispatchTexture>,
        >,
    ) {
        // The underlying backends (Metal, Vulkan, etc.) handle resource transitions
        // automatically; no explicit API call is needed.
    }
}

// ── CRenderBundleEncoder ──────────────────────────────────────────────────────

pub struct CRenderBundleEncoder {
    pub(crate) ptr: native::WGPURenderBundleEncoder,
}
impl std::fmt::Debug for CRenderBundleEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CRenderBundleEncoder")
            .field("ptr", &self.ptr)
            .finish()
    }
}
unsafe impl Send for CRenderBundleEncoder {}
unsafe impl Sync for CRenderBundleEncoder {}

impl Drop for CRenderBundleEncoder {
    fn drop(&mut self) {
        unsafe { wgpuRenderBundleEncoderRelease(self.ptr) };
    }
}

impl RenderBundleEncoderInterface for CRenderBundleEncoder {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) {
        let pp_ptr = pipeline.as_custom::<CRenderPipeline>().unwrap().ptr;
        unsafe { wgpuRenderBundleEncoderSetPipeline(self.ptr, pp_ptr) };
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let bg_ptr = bind_group
            .and_then(|bg| bg.as_custom::<CBindGroup>())
            .map(|bg| bg.ptr)
            .unwrap_or(std::ptr::null());
        unsafe {
            wgpuRenderBundleEncoderSetBindGroup(
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
        let buf_ptr = buffer.as_custom::<CBuffer>().unwrap().ptr;
        let c_size = size.map(|s| s.get()).unwrap_or(u64::MAX);
        unsafe {
            wgpuRenderBundleEncoderSetIndexBuffer(
                self.ptr,
                buf_ptr,
                conv::index_format_to_native(index_format),
                offset,
                c_size,
            )
        };
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&DispatchBuffer>,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        let buf_ptr = buffer
            .and_then(|b| b.as_custom::<CBuffer>())
            .map(|b| b.ptr)
            .unwrap_or(std::ptr::null());
        let c_size = size.map(|s| s.get()).unwrap_or(u64::MAX);
        unsafe { wgpuRenderBundleEncoderSetVertexBuffer(self.ptr, slot, buf_ptr, offset, c_size) };
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        unsafe {
            wgpuRenderBundleEncoderSetImmediates(
                self.ptr,
                offset,
                data.len() as u32,
                data.as_ptr().cast(),
            )
        };
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        unsafe {
            wgpuRenderBundleEncoderDraw(
                self.ptr,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        };
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        unsafe {
            wgpuRenderBundleEncoderDrawIndexed(
                self.ptr,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        };
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer.as_custom::<CBuffer>().unwrap().ptr;
        unsafe { wgpuRenderBundleEncoderDrawIndirect(self.ptr, buf_ptr, indirect_offset) };
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf_ptr = indirect_buffer.as_custom::<CBuffer>().unwrap().ptr;
        unsafe { wgpuRenderBundleEncoderDrawIndexedIndirect(self.ptr, buf_ptr, indirect_offset) };
    }

    fn finish(self, desc: &wgpu::RenderBundleDescriptor<'_>) -> DispatchRenderBundle
    where
        Self: Sized,
    {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPURenderBundleDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
        };
        let ptr = unsafe { wgpuRenderBundleEncoderFinish(self.ptr, Some(&c_desc)) };
        DispatchRenderBundle::custom(CRenderBundle { ptr })
    }

    fn finish_boxed(self: Box<Self>, desc: &wgpu::RenderBundleDescriptor<'_>) -> DispatchRenderBundle {
        (*self).finish(desc)
    }
}
