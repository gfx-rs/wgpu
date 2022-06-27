use super::conv;

use arrayvec::ArrayVec;
use ash::{extensions::ext, vk};
use inplace_it::inplace_or_alloc_from_iter;

use std::{mem, ops::Range, slice};

const ALLOCATION_GRANULARITY: u32 = 16;
const DST_IMAGE_LAYOUT: vk::ImageLayout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

impl super::Texture {
    fn map_buffer_copies<T>(&self, regions: T) -> impl Iterator<Item = vk::BufferImageCopy>
    where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let aspects = self.aspects;
        let fi = self.format_info;
        let copy_size = self.copy_size;
        regions.map(move |r| {
            let extent = r.texture_base.max_copy_size(&copy_size).min(&r.size);
            let (image_subresource, image_offset) =
                conv::map_subresource_layers(&r.texture_base, aspects);
            vk::BufferImageCopy {
                buffer_offset: r.buffer_layout.offset,
                buffer_row_length: r.buffer_layout.bytes_per_row.map_or(0, |bpr| {
                    fi.block_dimensions.0 as u32 * (bpr.get() / fi.block_size as u32)
                }),
                buffer_image_height: r
                    .buffer_layout
                    .rows_per_image
                    .map_or(0, |rpi| rpi.get() * fi.block_dimensions.1 as u32),
                image_subresource,
                image_offset,
                image_extent: conv::map_copy_extent(&extent),
            }
        })
    }
}

impl super::DeviceShared {
    fn debug_messenger(&self) -> Option<&ext::DebugUtils> {
        Some(&self.instance.debug_utils.as_ref()?.extension)
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        if self.free.is_empty() {
            let vk_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.raw)
                .command_buffer_count(ALLOCATION_GRANULARITY)
                .build();
            let cmd_buf_vec = self.device.raw.allocate_command_buffers(&vk_info)?;
            self.free.extend(cmd_buf_vec);
        }
        let raw = self.free.pop().unwrap();

        // Set the name unconditionally, since there might be a
        // previous name assigned to this.
        self.device.set_object_name(
            vk::ObjectType::COMMAND_BUFFER,
            raw,
            label.unwrap_or_default(),
        );

        // Reset this in case the last renderpass was never ended.
        self.rpass_debug_marker_active = false;

        let vk_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        self.device.raw.begin_command_buffer(raw, &vk_info)?;
        self.active = raw;

        Ok(())
    }

    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        let raw = self.active;
        self.active = vk::CommandBuffer::null();
        self.device.raw.end_command_buffer(raw)?;
        Ok(super::CommandBuffer { raw })
    }

    unsafe fn discard_encoding(&mut self) {
        self.discarded.push(self.active);
        self.active = vk::CommandBuffer::null();
    }

    unsafe fn reset_all<I>(&mut self, cmd_bufs: I)
    where
        I: Iterator<Item = super::CommandBuffer>,
    {
        self.temp.clear();
        self.free
            .extend(cmd_bufs.into_iter().map(|cmd_buf| cmd_buf.raw));
        self.free.append(&mut self.discarded);
        let _ = self
            .device
            .raw
            .reset_command_pool(self.raw, vk::CommandPoolResetFlags::RELEASE_RESOURCES);
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
        //Note: this is done so that we never end up with empty stage flags
        let mut src_stages = vk::PipelineStageFlags::TOP_OF_PIPE;
        let mut dst_stages = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        let vk_barriers = &mut self.temp.buffer_barriers;
        vk_barriers.clear();

        for bar in barriers {
            let (src_stage, src_access) = conv::map_buffer_usage_to_barrier(bar.usage.start);
            src_stages |= src_stage;
            let (dst_stage, dst_access) = conv::map_buffer_usage_to_barrier(bar.usage.end);
            dst_stages |= dst_stage;

            vk_barriers.push(
                vk::BufferMemoryBarrier::builder()
                    .buffer(bar.buffer.raw)
                    .size(vk::WHOLE_SIZE)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .build(),
            )
        }

        if !vk_barriers.is_empty() {
            self.device.raw.cmd_pipeline_barrier(
                self.active,
                src_stages,
                dst_stages,
                vk::DependencyFlags::empty(),
                &[],
                vk_barriers,
                &[],
            );
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        let mut src_stages = vk::PipelineStageFlags::empty();
        let mut dst_stages = vk::PipelineStageFlags::empty();
        let vk_barriers = &mut self.temp.image_barriers;
        vk_barriers.clear();

        for bar in barriers {
            let range = conv::map_subresource_range(&bar.range, bar.texture.aspects);
            let (src_stage, src_access) = conv::map_texture_usage_to_barrier(bar.usage.start);
            let src_layout = conv::derive_image_layout(bar.usage.start, bar.texture.aspects);
            src_stages |= src_stage;
            let (dst_stage, dst_access) = conv::map_texture_usage_to_barrier(bar.usage.end);
            let dst_layout = conv::derive_image_layout(bar.usage.end, bar.texture.aspects);
            dst_stages |= dst_stage;

            vk_barriers.push(
                vk::ImageMemoryBarrier::builder()
                    .image(bar.texture.raw)
                    .subresource_range(range)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .old_layout(src_layout)
                    .new_layout(dst_layout)
                    .build(),
            );
        }

        if !vk_barriers.is_empty() {
            self.device.raw.cmd_pipeline_barrier(
                self.active,
                src_stages,
                dst_stages,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                vk_barriers,
            );
        }
    }

    unsafe fn clear_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange) {
        self.device.raw.cmd_fill_buffer(
            self.active,
            buffer.raw,
            range.start,
            range.end - range.start,
            0,
        );
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let vk_regions_iter = regions.map(|r| vk::BufferCopy {
            src_offset: r.src_offset,
            dst_offset: r.dst_offset,
            size: r.size.get(),
        });

        inplace_or_alloc_from_iter(vk_regions_iter, |vk_regions| {
            self.device
                .raw
                .cmd_copy_buffer(self.active, src.raw, dst.raw, vk_regions)
        })
    }

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &super::Texture,
        src_usage: crate::TextureUses,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        let src_layout = conv::derive_image_layout(src_usage, src.aspects);

        let vk_regions_iter = regions.map(|r| {
            let (src_subresource, src_offset) =
                conv::map_subresource_layers(&r.src_base, src.aspects);
            let (dst_subresource, dst_offset) =
                conv::map_subresource_layers(&r.dst_base, dst.aspects);
            let extent = r
                .size
                .min(&r.src_base.max_copy_size(&src.copy_size))
                .min(&r.dst_base.max_copy_size(&dst.copy_size));
            vk::ImageCopy {
                src_subresource,
                src_offset,
                dst_subresource,
                dst_offset,
                extent: conv::map_copy_extent(&extent),
            }
        });

        inplace_or_alloc_from_iter(vk_regions_iter, |vk_regions| {
            self.device.raw.cmd_copy_image(
                self.active,
                src.raw,
                src_layout,
                dst.raw,
                DST_IMAGE_LAYOUT,
                vk_regions,
            );
        });
    }

    unsafe fn copy_buffer_to_texture<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let vk_regions_iter = dst.map_buffer_copies(regions);

        inplace_or_alloc_from_iter(vk_regions_iter, |vk_regions| {
            self.device.raw.cmd_copy_buffer_to_image(
                self.active,
                src.raw,
                dst.raw,
                DST_IMAGE_LAYOUT,
                vk_regions,
            );
        });
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &super::Texture,
        src_usage: crate::TextureUses,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let src_layout = conv::derive_image_layout(src_usage, src.aspects);
        let vk_regions_iter = src.map_buffer_copies(regions);

        inplace_or_alloc_from_iter(vk_regions_iter, |vk_regions| {
            self.device.raw.cmd_copy_image_to_buffer(
                self.active,
                src.raw,
                src_layout,
                dst.raw,
                vk_regions,
            );
        });
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        self.device.raw.cmd_begin_query(
            self.active,
            set.raw,
            index,
            vk::QueryControlFlags::empty(),
        );
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, index: u32) {
        self.device.raw.cmd_end_query(self.active, set.raw, index);
    }
    unsafe fn write_timestamp(&mut self, set: &super::QuerySet, index: u32) {
        self.device.raw.cmd_write_timestamp(
            self.active,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            set.raw,
            index,
        );
    }
    unsafe fn reset_queries(&mut self, set: &super::QuerySet, range: Range<u32>) {
        self.device.raw.cmd_reset_query_pool(
            self.active,
            set.raw,
            range.start,
            range.end - range.start,
        );
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
        self.device.raw.cmd_copy_query_pool_results(
            self.active,
            set.raw,
            range.start,
            range.end - range.start,
            buffer.raw,
            offset,
            stride.get(),
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
        let mut vk_clear_values =
            ArrayVec::<vk::ClearValue, { super::MAX_TOTAL_ATTACHMENTS }>::new();
        let mut vk_image_views = ArrayVec::<vk::ImageView, { super::MAX_TOTAL_ATTACHMENTS }>::new();
        let mut rp_key = super::RenderPassKey::default();
        let mut fb_key = super::FramebufferKey {
            attachments: ArrayVec::default(),
            extent: desc.extent,
            sample_count: desc.sample_count,
        };
        let caps = &self.device.private_caps;

        for cat in desc.color_attachments {
            if let Some(cat) = cat.as_ref() {
                vk_clear_values.push(vk::ClearValue {
                    color: cat.make_vk_clear_color(),
                });
                vk_image_views.push(cat.target.view.raw);
                let color = super::ColorAttachmentKey {
                    base: cat.target.make_attachment_key(cat.ops, caps),
                    resolve: cat.resolve_target.as_ref().map(|target| {
                        target.make_attachment_key(crate::AttachmentOps::STORE, caps)
                    }),
                };

                rp_key.colors.push(Some(color));
                fb_key.attachments.push(cat.target.view.attachment.clone());
                if let Some(ref at) = cat.resolve_target {
                    vk_clear_values.push(mem::zeroed());
                    vk_image_views.push(at.view.raw);
                    fb_key.attachments.push(at.view.attachment.clone());
                }

                // Assert this attachment is valid for the detected multiview, as a sanity check
                // The driver crash for this is really bad on AMD, so the check is worth it
                if let Some(multiview) = desc.multiview {
                    assert_eq!(cat.target.view.layers, multiview);
                    if let Some(ref resolve_target) = cat.resolve_target {
                        assert_eq!(resolve_target.view.layers, multiview);
                    }
                }
            } else {
                rp_key.colors.push(None);
            }
        }
        if let Some(ref ds) = desc.depth_stencil_attachment {
            vk_clear_values.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: ds.clear_value.0,
                    stencil: ds.clear_value.1,
                },
            });
            vk_image_views.push(ds.target.view.raw);
            rp_key.depth_stencil = Some(super::DepthStencilAttachmentKey {
                base: ds.target.make_attachment_key(ds.depth_ops, caps),
                stencil_ops: ds.stencil_ops,
            });
            fb_key.attachments.push(ds.target.view.attachment.clone());

            // Assert this attachment is valid for the detected multiview, as a sanity check
            // The driver crash for this is really bad on AMD, so the check is worth it
            if let Some(multiview) = desc.multiview {
                assert_eq!(ds.target.view.layers, multiview);
            }
        }
        rp_key.sample_count = fb_key.sample_count;
        rp_key.multiview = desc.multiview;

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: desc.extent.width,
                height: desc.extent.height,
            },
        };
        let vk_viewports = [vk::Viewport {
            x: 0.0,
            y: if self.device.private_caps.flip_y_requires_shift {
                desc.extent.height as f32
            } else {
                0.0
            },
            width: desc.extent.width as f32,
            height: -(desc.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let raw_pass = self.device.make_render_pass(rp_key).unwrap();
        let raw_framebuffer = self
            .device
            .make_framebuffer(fb_key, raw_pass, desc.label)
            .unwrap();

        let mut vk_info = vk::RenderPassBeginInfo::builder()
            .render_pass(raw_pass)
            .render_area(render_area)
            .clear_values(&vk_clear_values)
            .framebuffer(raw_framebuffer);
        let mut vk_attachment_info = if caps.imageless_framebuffers {
            Some(
                vk::RenderPassAttachmentBeginInfo::builder()
                    .attachments(&vk_image_views)
                    .build(),
            )
        } else {
            None
        };
        if let Some(attachment_info) = vk_attachment_info.as_mut() {
            vk_info = vk_info.push_next(attachment_info);
        }

        if let Some(label) = desc.label {
            self.begin_debug_marker(label);
            self.rpass_debug_marker_active = true;
        }

        self.device
            .raw
            .cmd_set_viewport(self.active, 0, &vk_viewports);
        self.device
            .raw
            .cmd_set_scissor(self.active, 0, &[render_area]);
        self.device
            .raw
            .cmd_begin_render_pass(self.active, &vk_info, vk::SubpassContents::INLINE);

        self.bind_point = vk::PipelineBindPoint::GRAPHICS;
    }
    unsafe fn end_render_pass(&mut self) {
        self.device.raw.cmd_end_render_pass(self.active);
        if self.rpass_debug_marker_active {
            self.end_debug_marker();
            self.rpass_debug_marker_active = false;
        }
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let sets = [*group.set.raw()];
        self.device.raw.cmd_bind_descriptor_sets(
            self.active,
            self.bind_point,
            layout.raw,
            index,
            &sets,
            dynamic_offsets,
        );
    }
    unsafe fn set_push_constants(
        &mut self,
        layout: &super::PipelineLayout,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u32],
    ) {
        self.device.raw.cmd_push_constants(
            self.active,
            layout.raw,
            conv::map_shader_stage(stages),
            offset,
            slice::from_raw_parts(data.as_ptr() as _, data.len() * 4),
        );
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        if let Some(ext) = self.device.debug_messenger() {
            let cstr = self.temp.make_c_str(label);
            let vk_label = vk::DebugUtilsLabelEXT::builder().label_name(cstr).build();
            ext.cmd_insert_debug_utils_label(self.active, &vk_label);
        }
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        if let Some(ext) = self.device.debug_messenger() {
            let cstr = self.temp.make_c_str(group_label);
            let vk_label = vk::DebugUtilsLabelEXT::builder().label_name(cstr).build();
            ext.cmd_begin_debug_utils_label(self.active, &vk_label);
        }
    }
    unsafe fn end_debug_marker(&mut self) {
        if let Some(ext) = self.device.debug_messenger() {
            ext.cmd_end_debug_utils_label(self.active);
        }
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        self.device.raw.cmd_bind_pipeline(
            self.active,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.raw,
        );
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
        self.device.raw.cmd_bind_index_buffer(
            self.active,
            binding.buffer.raw,
            binding.offset,
            conv::map_index_format(format),
        );
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
        let vk_buffers = [binding.buffer.raw];
        let vk_offsets = [binding.offset];
        self.device
            .raw
            .cmd_bind_vertex_buffers(self.active, index, &vk_buffers, &vk_offsets);
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let vk_viewports = [vk::Viewport {
            x: rect.x,
            y: if self.device.private_caps.flip_y_requires_shift {
                rect.y + rect.h
            } else {
                rect.y
            },
            width: rect.w,
            height: -rect.h, // flip Y
            min_depth: depth_range.start,
            max_depth: depth_range.end,
        }];
        self.device
            .raw
            .cmd_set_viewport(self.active, 0, &vk_viewports);
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        let vk_scissors = [vk::Rect2D {
            offset: vk::Offset2D {
                x: rect.x as i32,
                y: rect.y as i32,
            },
            extent: vk::Extent2D {
                width: rect.w,
                height: rect.h,
            },
        }];
        self.device
            .raw
            .cmd_set_scissor(self.active, 0, &vk_scissors);
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.device.raw.cmd_set_stencil_reference(
            self.active,
            vk::StencilFaceFlags::FRONT_AND_BACK,
            value,
        );
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        self.device.raw.cmd_set_blend_constants(self.active, color);
    }

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.device.raw.cmd_draw(
            self.active,
            vertex_count,
            instance_count,
            start_vertex,
            start_instance,
        );
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.device.raw.cmd_draw_indexed(
            self.active,
            index_count,
            instance_count,
            start_index,
            base_vertex,
            start_instance,
        );
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.device.raw.cmd_draw_indirect(
            self.active,
            buffer.raw,
            offset,
            draw_count,
            mem::size_of::<wgt::DrawIndirectArgs>() as u32,
        );
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.device.raw.cmd_draw_indexed_indirect(
            self.active,
            buffer.raw,
            offset,
            draw_count,
            mem::size_of::<wgt::DrawIndexedIndirectArgs>() as u32,
        );
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        let stride = mem::size_of::<wgt::DrawIndirectArgs>() as u32;
        match self.device.extension_fns.draw_indirect_count {
            Some(super::ExtensionFn::Extension(ref t)) => {
                t.cmd_draw_indirect_count(
                    self.active,
                    buffer.raw,
                    offset,
                    count_buffer.raw,
                    count_offset,
                    max_count,
                    stride,
                );
            }
            Some(super::ExtensionFn::Promoted) => {
                self.device.raw.cmd_draw_indirect_count(
                    self.active,
                    buffer.raw,
                    offset,
                    count_buffer.raw,
                    count_offset,
                    max_count,
                    stride,
                );
            }
            None => panic!("Feature `DRAW_INDIRECT_COUNT` not enabled"),
        }
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        let stride = mem::size_of::<wgt::DrawIndexedIndirectArgs>() as u32;
        match self.device.extension_fns.draw_indirect_count {
            Some(super::ExtensionFn::Extension(ref t)) => {
                t.cmd_draw_indexed_indirect_count(
                    self.active,
                    buffer.raw,
                    offset,
                    count_buffer.raw,
                    count_offset,
                    max_count,
                    stride,
                );
            }
            Some(super::ExtensionFn::Promoted) => {
                self.device.raw.cmd_draw_indexed_indirect_count(
                    self.active,
                    buffer.raw,
                    offset,
                    count_buffer.raw,
                    count_offset,
                    max_count,
                    stride,
                );
            }
            None => panic!("Feature `DRAW_INDIRECT_COUNT` not enabled"),
        }
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {
        self.bind_point = vk::PipelineBindPoint::COMPUTE;
        if let Some(label) = desc.label {
            self.begin_debug_marker(label);
            self.rpass_debug_marker_active = true;
        }
    }
    unsafe fn end_compute_pass(&mut self) {
        if self.rpass_debug_marker_active {
            self.end_debug_marker();
            self.rpass_debug_marker_active = false
        }
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {
        self.device.raw.cmd_bind_pipeline(
            self.active,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.raw,
        );
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        self.device
            .raw
            .cmd_dispatch(self.active, count[0], count[1], count[2]);
    }
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.device
            .raw
            .cmd_dispatch_indirect(self.active, buffer.raw, offset)
    }
}

#[test]
fn check_dst_image_layout() {
    assert_eq!(
        conv::derive_image_layout(crate::TextureUses::COPY_DST, crate::FormatAspects::empty()),
        DST_IMAGE_LAYOUT
    );
}
