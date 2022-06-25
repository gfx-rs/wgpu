use super::{conv, AsNative};
use std::{mem, ops::Range};

// has to match `Temp::binding_sizes`
const WORD_SIZE: usize = 4;

impl Default for super::CommandState {
    fn default() -> Self {
        Self {
            blit: None,
            render: None,
            compute: None,
            raw_primitive_type: mtl::MTLPrimitiveType::Point,
            index: None,
            raw_wg_size: mtl::MTLSize::new(0, 0, 0),
            stage_infos: Default::default(),
            storage_buffer_length_map: Default::default(),
            work_group_memory_sizes: Vec::new(),
            push_constants: Vec::new(),
        }
    }
}

impl super::CommandEncoder {
    fn enter_blit(&mut self) -> &mtl::BlitCommandEncoderRef {
        if self.state.blit.is_none() {
            debug_assert!(self.state.render.is_none() && self.state.compute.is_none());
            objc::rc::autoreleasepool(|| {
                let cmd_buf = self.raw_cmd_buf.as_ref().unwrap();
                self.state.blit = Some(cmd_buf.new_blit_command_encoder().to_owned());
            });
        }
        self.state.blit.as_ref().unwrap()
    }

    pub(super) fn leave_blit(&mut self) {
        if let Some(encoder) = self.state.blit.take() {
            encoder.end_encoding();
        }
    }

    fn enter_any(&mut self) -> Option<&mtl::CommandEncoderRef> {
        if let Some(ref encoder) = self.state.render {
            Some(encoder)
        } else if let Some(ref encoder) = self.state.compute {
            Some(encoder)
        } else if let Some(ref encoder) = self.state.blit {
            Some(encoder)
        } else {
            None
        }
    }

    fn begin_pass(&mut self) {
        self.state.reset();
        self.leave_blit();
    }
}

impl super::CommandState {
    fn reset(&mut self) {
        self.storage_buffer_length_map.clear();
        self.stage_infos.vs.clear();
        self.stage_infos.fs.clear();
        self.stage_infos.cs.clear();
        self.work_group_memory_sizes.clear();
        self.push_constants.clear();
    }

    fn make_sizes_buffer_update<'a>(
        &self,
        stage: naga::ShaderStage,
        result_sizes: &'a mut Vec<u32>,
    ) -> Option<(u32, &'a [u32])> {
        let stage_info = &self.stage_infos[stage];
        let slot = stage_info.sizes_slot?;

        result_sizes.clear();
        result_sizes.extend(
            stage_info
                .sized_bindings
                .iter()
                .filter_map(|br| self.storage_buffer_length_map.get(br))
                .map(|size| size.get().min(!0u32 as u64) as u32),
        );

        if !result_sizes.is_empty() {
            Some((slot as _, result_sizes))
        } else {
            None
        }
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        let queue = &self.raw_queue.lock();
        let retain_references = self.shared.settings.retain_command_buffer_references;
        let raw = objc::rc::autoreleasepool(move || {
            let cmd_buf_ref = if retain_references {
                queue.new_command_buffer()
            } else {
                queue.new_command_buffer_with_unretained_references()
            };
            cmd_buf_ref.to_owned()
        });

        if let Some(label) = label {
            raw.set_label(label);
        }
        self.raw_cmd_buf = Some(raw);

        Ok(())
    }

    unsafe fn discard_encoding(&mut self) {
        self.leave_blit();
        // when discarding, we don't have a guarantee that
        // everything is in a good state, so check carefully
        if let Some(encoder) = self.state.render.take() {
            encoder.end_encoding();
        }
        if let Some(encoder) = self.state.compute.take() {
            encoder.end_encoding();
        }
        self.raw_cmd_buf = None;
    }

    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        self.leave_blit();
        assert!(self.state.render.is_none());
        assert!(self.state.compute.is_none());
        Ok(super::CommandBuffer {
            raw: self.raw_cmd_buf.take().unwrap(),
        })
    }

    unsafe fn reset_all<I>(&mut self, _cmd_bufs: I)
    where
        I: Iterator<Item = super::CommandBuffer>,
    {
        //do nothing
    }

    unsafe fn transition_buffers<'a, T>(&mut self, _barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
    }

    unsafe fn transition_textures<'a, T>(&mut self, _barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
    }

    unsafe fn clear_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange) {
        let encoder = self.enter_blit();
        encoder.fill_buffer(&buffer.raw, conv::map_range(&range), 0);
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let encoder = self.enter_blit();
        for copy in regions {
            encoder.copy_from_buffer(
                &src.raw,
                copy.src_offset,
                &dst.raw,
                copy.dst_offset,
                copy.size.get(),
            );
        }
    }

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUses,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        let encoder = self.enter_blit();
        for copy in regions {
            let src_origin = conv::map_origin(&copy.src_base.origin);
            let dst_origin = conv::map_origin(&copy.dst_base.origin);
            // no clamping is done: Metal expects physical sizes here
            let extent = conv::map_copy_extent(&copy.size);
            encoder.copy_from_texture(
                &src.raw,
                copy.src_base.array_layer as u64,
                copy.src_base.mip_level as u64,
                src_origin,
                extent,
                &dst.raw,
                copy.dst_base.array_layer as u64,
                copy.dst_base.mip_level as u64,
                dst_origin,
            );
        }
    }

    unsafe fn copy_buffer_to_texture<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let encoder = self.enter_blit();
        for copy in regions {
            let dst_origin = conv::map_origin(&copy.texture_base.origin);
            // Metal expects buffer-texture copies in virtual sizes
            let extent = copy
                .texture_base
                .max_copy_size(&dst.copy_size)
                .min(&copy.size);
            let bytes_per_row = copy
                .buffer_layout
                .bytes_per_row
                .map_or(0, |v| v.get() as u64);
            let bytes_per_image = copy
                .buffer_layout
                .rows_per_image
                .map_or(0, |v| v.get() as u64 * bytes_per_row);
            encoder.copy_from_buffer_to_texture(
                &src.raw,
                copy.buffer_layout.offset,
                bytes_per_row,
                bytes_per_image,
                conv::map_copy_extent(&extent),
                &dst.raw,
                copy.texture_base.array_layer as u64,
                copy.texture_base.mip_level as u64,
                dst_origin,
                mtl::MTLBlitOption::empty(),
            );
        }
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUses,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let encoder = self.enter_blit();
        for copy in regions {
            let src_origin = conv::map_origin(&copy.texture_base.origin);
            // Metal expects texture-buffer copies in virtual sizes
            let extent = copy
                .texture_base
                .max_copy_size(&src.copy_size)
                .min(&copy.size);
            let bytes_per_row = copy
                .buffer_layout
                .bytes_per_row
                .map_or(0, |v| v.get() as u64);
            let bytes_per_image = copy
                .buffer_layout
                .rows_per_image
                .map_or(0, |v| v.get() as u64 * bytes_per_row);
            encoder.copy_from_texture_to_buffer(
                &src.raw,
                copy.texture_base.array_layer as u64,
                copy.texture_base.mip_level as u64,
                src_origin,
                conv::map_copy_extent(&extent),
                &dst.raw,
                copy.buffer_layout.offset,
                bytes_per_row,
                bytes_per_image,
                mtl::MTLBlitOption::empty(),
            );
        }
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        match set.ty {
            wgt::QueryType::Occlusion => {
                self.state
                    .render
                    .as_ref()
                    .unwrap()
                    .set_visibility_result_mode(
                        mtl::MTLVisibilityResultMode::Boolean,
                        index as u64 * crate::QUERY_SIZE,
                    );
            }
            _ => {}
        }
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, _index: u32) {
        match set.ty {
            wgt::QueryType::Occlusion => {
                self.state
                    .render
                    .as_ref()
                    .unwrap()
                    .set_visibility_result_mode(mtl::MTLVisibilityResultMode::Disabled, 0);
            }
            _ => {}
        }
    }
    unsafe fn write_timestamp(&mut self, _set: &super::QuerySet, _index: u32) {}
    unsafe fn reset_queries(&mut self, set: &super::QuerySet, range: Range<u32>) {
        let encoder = self.enter_blit();
        let raw_range = mtl::NSRange {
            location: range.start as u64 * crate::QUERY_SIZE,
            length: (range.end - range.start) as u64 * crate::QUERY_SIZE,
        };
        encoder.fill_buffer(&set.raw_buffer, raw_range, 0);
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        _: wgt::BufferSize, // Metal doesn't support queries that are bigger than a single element are not supported
    ) {
        let encoder = self.enter_blit();
        let size = (range.end - range.start) as u64 * crate::QUERY_SIZE;
        encoder.copy_from_buffer(
            &set.raw_buffer,
            range.start as u64 * crate::QUERY_SIZE,
            &buffer.raw,
            offset,
            size,
        );
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
        self.begin_pass();
        self.state.index = None;

        objc::rc::autoreleasepool(|| {
            let descriptor = mtl::RenderPassDescriptor::new();
            //TODO: set visibility results buffer

            for (i, at) in desc.color_attachments.iter().enumerate() {
                if let Some(at) = at.as_ref() {
                    let at_descriptor = descriptor.color_attachments().object_at(i as u64).unwrap();
                    at_descriptor.set_texture(Some(&at.target.view.raw));
                    if let Some(ref resolve) = at.resolve_target {
                        //Note: the selection of levels and slices is already handled by `TextureView`
                        at_descriptor.set_resolve_texture(Some(&resolve.view.raw));
                    }
                    let load_action = if at.ops.contains(crate::AttachmentOps::LOAD) {
                        mtl::MTLLoadAction::Load
                    } else {
                        at_descriptor.set_clear_color(conv::map_clear_color(&at.clear_value));
                        mtl::MTLLoadAction::Clear
                    };
                    let store_action = conv::map_store_action(
                        at.ops.contains(crate::AttachmentOps::STORE),
                        at.resolve_target.is_some(),
                    );
                    at_descriptor.set_load_action(load_action);
                    at_descriptor.set_store_action(store_action);
                }
            }

            if let Some(ref at) = desc.depth_stencil_attachment {
                if at.target.view.aspects.contains(crate::FormatAspects::DEPTH) {
                    let at_descriptor = descriptor.depth_attachment().unwrap();
                    at_descriptor.set_texture(Some(&at.target.view.raw));

                    let load_action = if at.depth_ops.contains(crate::AttachmentOps::LOAD) {
                        mtl::MTLLoadAction::Load
                    } else {
                        at_descriptor.set_clear_depth(at.clear_value.0 as f64);
                        mtl::MTLLoadAction::Clear
                    };
                    let store_action = if at.depth_ops.contains(crate::AttachmentOps::STORE) {
                        mtl::MTLStoreAction::Store
                    } else {
                        mtl::MTLStoreAction::DontCare
                    };
                    at_descriptor.set_load_action(load_action);
                    at_descriptor.set_store_action(store_action);
                }
                if at
                    .target
                    .view
                    .aspects
                    .contains(crate::FormatAspects::STENCIL)
                {
                    let at_descriptor = descriptor.stencil_attachment().unwrap();
                    at_descriptor.set_texture(Some(&at.target.view.raw));

                    let load_action = if at.stencil_ops.contains(crate::AttachmentOps::LOAD) {
                        mtl::MTLLoadAction::Load
                    } else {
                        at_descriptor.set_clear_stencil(at.clear_value.1);
                        mtl::MTLLoadAction::Clear
                    };
                    let store_action = if at.stencil_ops.contains(crate::AttachmentOps::STORE) {
                        mtl::MTLStoreAction::Store
                    } else {
                        mtl::MTLStoreAction::DontCare
                    };
                    at_descriptor.set_load_action(load_action);
                    at_descriptor.set_store_action(store_action);
                }
            }

            let raw = self.raw_cmd_buf.as_ref().unwrap();
            let encoder = raw.new_render_command_encoder(descriptor);
            if let Some(label) = desc.label {
                encoder.set_label(label);
            }
            self.state.render = Some(encoder.to_owned());
        });
    }

    unsafe fn end_render_pass(&mut self) {
        self.state.render.take().unwrap().end_encoding();
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        group_index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let bg_info = &layout.bind_group_infos[group_index as usize];

        if let Some(ref encoder) = self.state.render {
            let mut changes_sizes_buffer = false;
            for index in 0..group.counters.vs.buffers {
                let buf = &group.buffers[index as usize];
                let mut offset = buf.offset;
                if let Some(dyn_index) = buf.dynamic_index {
                    offset += dynamic_offsets[dyn_index as usize] as wgt::BufferAddress;
                }
                encoder.set_vertex_buffer(
                    (bg_info.base_resource_indices.vs.buffers + index) as u64,
                    Some(buf.ptr.as_native()),
                    offset,
                );
                if let Some(size) = buf.binding_size {
                    let br = naga::ResourceBinding {
                        group: group_index,
                        binding: buf.binding_location,
                    };
                    self.state.storage_buffer_length_map.insert(br, size);
                    changes_sizes_buffer = true;
                }
            }
            if changes_sizes_buffer {
                if let Some((index, sizes)) = self.state.make_sizes_buffer_update(
                    naga::ShaderStage::Vertex,
                    &mut self.temp.binding_sizes,
                ) {
                    encoder.set_vertex_bytes(
                        index as _,
                        (sizes.len() * WORD_SIZE) as u64,
                        sizes.as_ptr() as _,
                    );
                }
            }

            changes_sizes_buffer = false;
            for index in 0..group.counters.fs.buffers {
                let buf = &group.buffers[(group.counters.vs.buffers + index) as usize];
                let mut offset = buf.offset;
                if let Some(dyn_index) = buf.dynamic_index {
                    offset += dynamic_offsets[dyn_index as usize] as wgt::BufferAddress;
                }
                encoder.set_fragment_buffer(
                    (bg_info.base_resource_indices.fs.buffers + index) as u64,
                    Some(buf.ptr.as_native()),
                    offset,
                );
                if let Some(size) = buf.binding_size {
                    let br = naga::ResourceBinding {
                        group: group_index,
                        binding: buf.binding_location,
                    };
                    self.state.storage_buffer_length_map.insert(br, size);
                    changes_sizes_buffer = true;
                }
            }
            if changes_sizes_buffer {
                if let Some((index, sizes)) = self.state.make_sizes_buffer_update(
                    naga::ShaderStage::Fragment,
                    &mut self.temp.binding_sizes,
                ) {
                    encoder.set_fragment_bytes(
                        index as _,
                        (sizes.len() * WORD_SIZE) as u64,
                        sizes.as_ptr() as _,
                    );
                }
            }

            for index in 0..group.counters.vs.samplers {
                let res = group.samplers[index as usize];
                encoder.set_vertex_sampler_state(
                    (bg_info.base_resource_indices.vs.samplers + index) as u64,
                    Some(res.as_native()),
                );
            }
            for index in 0..group.counters.fs.samplers {
                let res = group.samplers[(group.counters.vs.samplers + index) as usize];
                encoder.set_fragment_sampler_state(
                    (bg_info.base_resource_indices.fs.samplers + index) as u64,
                    Some(res.as_native()),
                );
            }

            for index in 0..group.counters.vs.textures {
                let res = group.textures[index as usize];
                encoder.set_vertex_texture(
                    (bg_info.base_resource_indices.vs.textures + index) as u64,
                    Some(res.as_native()),
                );
            }
            for index in 0..group.counters.fs.textures {
                let res = group.textures[(group.counters.vs.textures + index) as usize];
                encoder.set_fragment_texture(
                    (bg_info.base_resource_indices.fs.textures + index) as u64,
                    Some(res.as_native()),
                );
            }
        }

        if let Some(ref encoder) = self.state.compute {
            let index_base = super::ResourceData {
                buffers: group.counters.vs.buffers + group.counters.fs.buffers,
                samplers: group.counters.vs.samplers + group.counters.fs.samplers,
                textures: group.counters.vs.textures + group.counters.fs.textures,
            };

            let mut changes_sizes_buffer = false;
            for index in 0..group.counters.cs.buffers {
                let buf = &group.buffers[(index_base.buffers + index) as usize];
                let mut offset = buf.offset;
                if let Some(dyn_index) = buf.dynamic_index {
                    offset += dynamic_offsets[dyn_index as usize] as wgt::BufferAddress;
                }
                encoder.set_buffer(
                    (bg_info.base_resource_indices.cs.buffers + index) as u64,
                    Some(buf.ptr.as_native()),
                    offset,
                );
                if let Some(size) = buf.binding_size {
                    let br = naga::ResourceBinding {
                        group: group_index,
                        binding: buf.binding_location,
                    };
                    self.state.storage_buffer_length_map.insert(br, size);
                    changes_sizes_buffer = true;
                }
            }
            if changes_sizes_buffer {
                if let Some((index, sizes)) = self.state.make_sizes_buffer_update(
                    naga::ShaderStage::Compute,
                    &mut self.temp.binding_sizes,
                ) {
                    encoder.set_bytes(
                        index as _,
                        (sizes.len() * WORD_SIZE) as u64,
                        sizes.as_ptr() as _,
                    );
                }
            }

            for index in 0..group.counters.cs.samplers {
                let res = group.samplers[(index_base.samplers + index) as usize];
                encoder.set_sampler_state(
                    (bg_info.base_resource_indices.cs.samplers + index) as u64,
                    Some(res.as_native()),
                );
            }
            for index in 0..group.counters.cs.textures {
                let res = group.textures[(index_base.textures + index) as usize];
                encoder.set_texture(
                    (bg_info.base_resource_indices.cs.textures + index) as u64,
                    Some(res.as_native()),
                );
            }
        }
    }

    unsafe fn set_push_constants(
        &mut self,
        layout: &super::PipelineLayout,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u32],
    ) {
        let state_pc = &mut self.state.push_constants;
        if state_pc.len() < layout.total_push_constants as usize {
            state_pc.resize(layout.total_push_constants as usize, 0);
        }
        assert_eq!(offset as usize % WORD_SIZE, 0);

        let offset = offset as usize / WORD_SIZE;
        state_pc[offset..offset + data.len()].copy_from_slice(data);

        if stages.contains(wgt::ShaderStages::COMPUTE) {
            self.state.compute.as_ref().unwrap().set_bytes(
                layout.push_constants_infos.cs.unwrap().buffer_index as _,
                (layout.total_push_constants as usize * WORD_SIZE) as _,
                state_pc.as_ptr() as _,
            )
        }
        if stages.contains(wgt::ShaderStages::VERTEX) {
            self.state.render.as_ref().unwrap().set_vertex_bytes(
                layout.push_constants_infos.vs.unwrap().buffer_index as _,
                (layout.total_push_constants as usize * WORD_SIZE) as _,
                state_pc.as_ptr() as _,
            )
        }
        if stages.contains(wgt::ShaderStages::FRAGMENT) {
            self.state.render.as_ref().unwrap().set_fragment_bytes(
                layout.push_constants_infos.fs.unwrap().buffer_index as _,
                (layout.total_push_constants as usize * WORD_SIZE) as _,
                state_pc.as_ptr() as _,
            )
        }
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        if let Some(encoder) = self.enter_any() {
            encoder.insert_debug_signpost(label);
        }
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        if let Some(encoder) = self.enter_any() {
            encoder.push_debug_group(group_label);
        } else if let Some(ref buf) = self.raw_cmd_buf {
            buf.push_debug_group(group_label);
        }
    }
    unsafe fn end_debug_marker(&mut self) {
        if let Some(encoder) = self.enter_any() {
            encoder.pop_debug_group();
        } else if let Some(ref buf) = self.raw_cmd_buf {
            buf.pop_debug_group();
        }
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        self.state.raw_primitive_type = pipeline.raw_primitive_type;
        self.state.stage_infos.vs.assign_from(&pipeline.vs_info);
        self.state.stage_infos.fs.assign_from(&pipeline.fs_info);

        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_render_pipeline_state(&pipeline.raw);
        encoder.set_front_facing_winding(pipeline.raw_front_winding);
        encoder.set_cull_mode(pipeline.raw_cull_mode);
        encoder.set_triangle_fill_mode(pipeline.raw_triangle_fill_mode);
        if let Some(depth_clip) = pipeline.raw_depth_clip_mode {
            encoder.set_depth_clip_mode(depth_clip);
        }
        if let Some((ref state, bias)) = pipeline.depth_stencil {
            encoder.set_depth_stencil_state(state);
            encoder.set_depth_bias(bias.constant as f32, bias.slope_scale, bias.clamp);
        }

        {
            if let Some((index, sizes)) = self
                .state
                .make_sizes_buffer_update(naga::ShaderStage::Vertex, &mut self.temp.binding_sizes)
            {
                encoder.set_vertex_bytes(
                    index as _,
                    (sizes.len() * WORD_SIZE) as u64,
                    sizes.as_ptr() as _,
                );
            }
        }
        if pipeline.fs_lib.is_some() {
            if let Some((index, sizes)) = self
                .state
                .make_sizes_buffer_update(naga::ShaderStage::Fragment, &mut self.temp.binding_sizes)
            {
                encoder.set_fragment_bytes(
                    index as _,
                    (sizes.len() * WORD_SIZE) as u64,
                    sizes.as_ptr() as _,
                );
            }
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
        let (stride, raw_type) = match format {
            wgt::IndexFormat::Uint16 => (2, mtl::MTLIndexType::UInt16),
            wgt::IndexFormat::Uint32 => (4, mtl::MTLIndexType::UInt32),
        };
        self.state.index = Some(super::IndexState {
            buffer_ptr: AsNative::from(binding.buffer.raw.as_ref()),
            offset: binding.offset,
            stride,
            raw_type,
        });
    }

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
        let buffer_index = self.shared.private_caps.max_buffers_per_stage as u64 - 1 - index as u64;
        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_vertex_buffer(buffer_index, Some(&binding.buffer.raw), binding.offset);
    }

    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let zfar = if self.shared.disabilities.broken_viewport_near_depth {
            depth_range.end - depth_range.start
        } else {
            depth_range.end
        };
        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_viewport(mtl::MTLViewport {
            originX: rect.x as _,
            originY: rect.y as _,
            width: rect.w as _,
            height: rect.h as _,
            znear: depth_range.start as _,
            zfar: zfar as _,
        });
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        //TODO: support empty scissors by modifying the viewport
        let scissor = mtl::MTLScissorRect {
            x: rect.x as _,
            y: rect.y as _,
            width: rect.w as _,
            height: rect.h as _,
        };
        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_scissor_rect(scissor);
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_stencil_front_back_reference_value(value, value);
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        let encoder = self.state.render.as_ref().unwrap();
        encoder.set_blend_color(color[0], color[1], color[2], color[3]);
    }

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        let encoder = self.state.render.as_ref().unwrap();
        if start_instance != 0 {
            encoder.draw_primitives_instanced_base_instance(
                self.state.raw_primitive_type,
                start_vertex as _,
                vertex_count as _,
                instance_count as _,
                start_instance as _,
            );
        } else if instance_count != 1 {
            encoder.draw_primitives_instanced(
                self.state.raw_primitive_type,
                start_vertex as _,
                vertex_count as _,
                instance_count as _,
            );
        } else {
            encoder.draw_primitives(
                self.state.raw_primitive_type,
                start_vertex as _,
                vertex_count as _,
            );
        }
    }

    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        let encoder = self.state.render.as_ref().unwrap();
        let index = self.state.index.as_ref().unwrap();
        let offset = index.offset + index.stride * start_index as wgt::BufferAddress;
        if base_vertex != 0 || start_instance != 0 {
            encoder.draw_indexed_primitives_instanced_base_instance(
                self.state.raw_primitive_type,
                index_count as _,
                index.raw_type,
                index.buffer_ptr.as_native(),
                offset,
                instance_count as _,
                base_vertex as _,
                start_instance as _,
            );
        } else if instance_count != 1 {
            encoder.draw_indexed_primitives_instanced(
                self.state.raw_primitive_type,
                index_count as _,
                index.raw_type,
                index.buffer_ptr.as_native(),
                offset,
                instance_count as _,
            );
        } else {
            encoder.draw_indexed_primitives(
                self.state.raw_primitive_type,
                index_count as _,
                index.raw_type,
                index.buffer_ptr.as_native(),
                offset,
            );
        }
    }

    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        mut offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        let encoder = self.state.render.as_ref().unwrap();
        for _ in 0..draw_count {
            encoder.draw_primitives_indirect(self.state.raw_primitive_type, &buffer.raw, offset);
            offset += mem::size_of::<wgt::DrawIndirectArgs>() as wgt::BufferAddress;
        }
    }

    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        mut offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        let encoder = self.state.render.as_ref().unwrap();
        let index = self.state.index.as_ref().unwrap();
        for _ in 0..draw_count {
            encoder.draw_indexed_primitives_indirect(
                self.state.raw_primitive_type,
                index.raw_type,
                index.buffer_ptr.as_native(),
                index.offset,
                &buffer.raw,
                offset,
            );
            offset += mem::size_of::<wgt::DrawIndexedIndirectArgs>() as wgt::BufferAddress;
        }
    }

    unsafe fn draw_indirect_count(
        &mut self,
        _buffer: &super::Buffer,
        _offset: wgt::BufferAddress,
        _count_buffer: &super::Buffer,
        _count_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        //TODO
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        _buffer: &super::Buffer,
        _offset: wgt::BufferAddress,
        _count_buffer: &super::Buffer,
        _count_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        //TODO
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {
        self.begin_pass();

        let raw = self.raw_cmd_buf.as_ref().unwrap();
        let encoder = raw.new_compute_command_encoder();
        if let Some(label) = desc.label {
            encoder.set_label(label);
        }
        self.state.compute = Some(encoder.to_owned());
    }
    unsafe fn end_compute_pass(&mut self) {
        self.state.compute.take().unwrap().end_encoding();
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {
        self.state.raw_wg_size = pipeline.work_group_size;
        self.state.stage_infos.cs.assign_from(&pipeline.cs_info);

        let encoder = self.state.compute.as_ref().unwrap();
        encoder.set_compute_pipeline_state(&pipeline.raw);

        if let Some((index, sizes)) = self
            .state
            .make_sizes_buffer_update(naga::ShaderStage::Compute, &mut self.temp.binding_sizes)
        {
            encoder.set_bytes(
                index as _,
                (sizes.len() * WORD_SIZE) as u64,
                sizes.as_ptr() as _,
            );
        }

        // update the threadgroup memory sizes
        while self.state.work_group_memory_sizes.len() < pipeline.work_group_memory_sizes.len() {
            self.state.work_group_memory_sizes.push(0);
        }
        for (index, (cur_size, pipeline_size)) in self
            .state
            .work_group_memory_sizes
            .iter_mut()
            .zip(pipeline.work_group_memory_sizes.iter())
            .enumerate()
        {
            const ALIGN_MASK: u32 = 0xF; // must be a multiple of 16 bytes
            let size = ((*pipeline_size - 1) | ALIGN_MASK) + 1;
            if *cur_size != size {
                *cur_size = size;
                encoder.set_threadgroup_memory_length(index as _, size as _);
            }
        }
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        let encoder = self.state.compute.as_ref().unwrap();
        let raw_count = mtl::MTLSize {
            width: count[0] as u64,
            height: count[1] as u64,
            depth: count[2] as u64,
        };
        encoder.dispatch_thread_groups(raw_count, self.state.raw_wg_size);
    }

    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        let encoder = self.state.compute.as_ref().unwrap();
        encoder.dispatch_thread_groups_indirect(&buffer.raw, offset, self.state.raw_wg_size);
    }
}
