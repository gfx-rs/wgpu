use super::{conv, Command as C};
use arrayvec::ArrayVec;
use std::{mem, ops::Range};

#[derive(Clone, Copy, Debug, Default)]
struct TextureSlotDesc {
    tex_target: super::BindTarget,
    sampler_index: Option<u8>,
}

#[derive(Default)]
pub(super) struct State {
    topology: u32,
    primitive: super::PrimitiveState,
    index_format: wgt::IndexFormat,
    index_offset: wgt::BufferAddress,
    vertex_buffers:
        [(super::VertexBufferDesc, Option<super::BufferBinding>); crate::MAX_VERTEX_BUFFERS],
    vertex_attributes: ArrayVec<super::AttributeDesc, { super::MAX_VERTEX_ATTRIBUTES }>,
    color_targets: ArrayVec<super::ColorTargetDesc, { crate::MAX_COLOR_ATTACHMENTS }>,
    stencil: super::StencilState,
    depth_bias: wgt::DepthBiasState,
    alpha_to_coverage_enabled: bool,
    samplers: [Option<glow::Sampler>; super::MAX_SAMPLERS],
    texture_slots: [TextureSlotDesc; super::MAX_TEXTURE_SLOTS],
    render_size: wgt::Extent3d,
    resolve_attachments: ArrayVec<(u32, super::TextureView), { crate::MAX_COLOR_ATTACHMENTS }>,
    invalidate_attachments: ArrayVec<u32, { crate::MAX_COLOR_ATTACHMENTS + 2 }>,
    has_pass_label: bool,
    instance_vbuf_mask: usize,
    dirty_vbuf_mask: usize,
    active_first_instance: u32,
    push_offset_to_uniform: ArrayVec<super::UniformDesc, { super::MAX_PUSH_CONSTANTS }>,
}

impl super::CommandBuffer {
    fn clear(&mut self) {
        self.label = None;
        self.commands.clear();
        self.data_bytes.clear();
        self.queries.clear();
    }

    fn add_marker(&mut self, marker: &str) -> Range<u32> {
        let start = self.data_bytes.len() as u32;
        self.data_bytes.extend(marker.as_bytes());
        start..self.data_bytes.len() as u32
    }

    fn add_push_constant_data(&mut self, data: &[u32]) -> Range<u32> {
        let data_raw = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const _,
                data.len() * mem::size_of::<u32>(),
            )
        };
        let start = self.data_bytes.len();
        assert!(start < u32::MAX as usize);
        self.data_bytes.extend_from_slice(data_raw);
        let end = self.data_bytes.len();
        assert!(end < u32::MAX as usize);
        (start as u32)..(end as u32)
    }
}

impl super::CommandEncoder {
    fn rebind_stencil_func(&mut self) {
        fn make(s: &super::StencilSide, face: u32) -> C {
            C::SetStencilFunc {
                face,
                function: s.function,
                reference: s.reference,
                read_mask: s.mask_read,
            }
        }

        let s = &self.state.stencil;
        if s.front.function == s.back.function
            && s.front.mask_read == s.back.mask_read
            && s.front.reference == s.back.reference
        {
            self.cmd_buffer
                .commands
                .push(make(&s.front, glow::FRONT_AND_BACK));
        } else {
            self.cmd_buffer.commands.push(make(&s.front, glow::FRONT));
            self.cmd_buffer.commands.push(make(&s.back, glow::BACK));
        }
    }

    fn rebind_vertex_data(&mut self, first_instance: u32) {
        if self
            .private_caps
            .contains(super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT)
        {
            for (index, pair) in self.state.vertex_buffers.iter().enumerate() {
                if self.state.dirty_vbuf_mask & (1 << index) == 0 {
                    continue;
                }
                let (buffer_desc, vb) = match *pair {
                    // Not all dirty bindings are necessarily filled. Some may be unused.
                    (_, None) => continue,
                    (ref vb_desc, Some(ref vb)) => (vb_desc.clone(), vb),
                };
                let instance_offset = match buffer_desc.step {
                    wgt::VertexStepMode::Vertex => 0,
                    wgt::VertexStepMode::Instance => first_instance * buffer_desc.stride,
                };

                self.cmd_buffer.commands.push(C::SetVertexBuffer {
                    index: index as u32,
                    buffer: super::BufferBinding {
                        raw: vb.raw,
                        offset: vb.offset + instance_offset as wgt::BufferAddress,
                    },
                    buffer_desc,
                });
                self.state.dirty_vbuf_mask ^= 1 << index;
            }
        } else {
            let mut vbuf_mask = 0;
            for attribute in self.state.vertex_attributes.iter() {
                if self.state.dirty_vbuf_mask & (1 << attribute.buffer_index) == 0 {
                    continue;
                }
                let (buffer_desc, vb) =
                    match self.state.vertex_buffers[attribute.buffer_index as usize] {
                        // Not all dirty bindings are necessarily filled. Some may be unused.
                        (_, None) => continue,
                        (ref vb_desc, Some(ref vb)) => (vb_desc.clone(), vb),
                    };

                let mut attribute_desc = attribute.clone();
                attribute_desc.offset += vb.offset as u32;
                if buffer_desc.step == wgt::VertexStepMode::Instance {
                    attribute_desc.offset += buffer_desc.stride * first_instance;
                }

                self.cmd_buffer.commands.push(C::SetVertexAttribute {
                    buffer: Some(vb.raw),
                    buffer_desc,
                    attribute_desc,
                });
                vbuf_mask |= 1 << attribute.buffer_index;
            }
            self.state.dirty_vbuf_mask ^= vbuf_mask;
        }
    }

    fn rebind_sampler_states(&mut self, dirty_textures: u32, dirty_samplers: u32) {
        for (texture_index, slot) in self.state.texture_slots.iter().enumerate() {
            if dirty_textures & (1 << texture_index) != 0
                || slot
                    .sampler_index
                    .map_or(false, |si| dirty_samplers & (1 << si) != 0)
            {
                let sampler = slot
                    .sampler_index
                    .and_then(|si| self.state.samplers[si as usize]);
                self.cmd_buffer
                    .commands
                    .push(C::BindSampler(texture_index as u32, sampler));
            }
        }
    }

    fn prepare_draw(&mut self, first_instance: u32) {
        if first_instance != self.state.active_first_instance {
            // rebind all per-instance buffers on first-instance change
            self.state.dirty_vbuf_mask |= self.state.instance_vbuf_mask;
            self.state.active_first_instance = first_instance;
        }
        if self.state.dirty_vbuf_mask != 0 {
            self.rebind_vertex_data(first_instance);
        }
    }

    fn set_pipeline_inner(&mut self, inner: &super::PipelineInner) {
        self.cmd_buffer.commands.push(C::SetProgram(inner.program));

        self.state.push_offset_to_uniform.clear();
        self.state
            .push_offset_to_uniform
            .extend(inner.uniforms.iter().cloned());

        // rebind textures, if needed
        let mut dirty_textures = 0u32;
        for (texture_index, (slot, &sampler_index)) in self
            .state
            .texture_slots
            .iter_mut()
            .zip(inner.sampler_map.iter())
            .enumerate()
        {
            if slot.sampler_index != sampler_index {
                slot.sampler_index = sampler_index;
                dirty_textures |= 1 << texture_index;
            }
        }
        if dirty_textures != 0 {
            self.rebind_sampler_states(dirty_textures, 0);
        }
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        self.state = State::default();
        self.cmd_buffer.label = label.map(str::to_string);
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        self.cmd_buffer.clear();
    }
    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        Ok(mem::take(&mut self.cmd_buffer))
    }
    unsafe fn reset_all<I>(&mut self, _command_buffers: I) {
        //TODO: could re-use the allocations in all these command buffers
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
        if !self
            .private_caps
            .contains(super::PrivateCapabilities::MEMORY_BARRIERS)
        {
            return;
        }
        for bar in barriers {
            // GLES only synchronizes storage -> anything explicitly
            if !bar
                .usage
                .start
                .contains(crate::BufferUses::STORAGE_READ_WRITE)
            {
                continue;
            }
            self.cmd_buffer
                .commands
                .push(C::BufferBarrier(bar.buffer.raw.unwrap(), bar.usage.end));
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        if !self
            .private_caps
            .contains(super::PrivateCapabilities::MEMORY_BARRIERS)
        {
            return;
        }

        let mut combined_usage = crate::TextureUses::empty();
        for bar in barriers {
            // GLES only synchronizes storage -> anything explicitly
            if !bar
                .usage
                .start
                .contains(crate::TextureUses::STORAGE_READ_WRITE)
            {
                continue;
            }
            // unlike buffers, there is no need for a concrete texture
            // object to be bound anywhere for a barrier
            combined_usage |= bar.usage.end;
        }

        if !combined_usage.is_empty() {
            self.cmd_buffer
                .commands
                .push(C::TextureBarrier(combined_usage));
        }
    }

    unsafe fn clear_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange) {
        self.cmd_buffer.commands.push(C::ClearBuffer {
            dst: buffer.clone(),
            dst_target: buffer.target,
            range,
        });
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let (src_target, dst_target) = if src.target == dst.target {
            (glow::COPY_READ_BUFFER, glow::COPY_WRITE_BUFFER)
        } else {
            (src.target, dst.target)
        };
        for copy in regions {
            self.cmd_buffer.commands.push(C::CopyBufferToBuffer {
                src: src.clone(),
                src_target,
                dst: dst.clone(),
                dst_target,
                copy,
            })
        }
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    unsafe fn copy_external_image_to_texture<T>(
        &mut self,
        src: &wgt::ImageCopyExternalImage,
        dst: &super::Texture,
        dst_premultiplication: bool,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        let (dst_raw, dst_target) = dst.inner.as_native();
        for copy in regions {
            self.cmd_buffer
                .commands
                .push(C::CopyExternalImageToTexture {
                    src: src.clone(),
                    dst: dst_raw,
                    dst_target,
                    dst_format: dst.format,
                    dst_premultiplication,
                    copy,
                })
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
        let (src_raw, src_target) = src.inner.as_native();
        let (dst_raw, dst_target) = dst.inner.as_native();
        for mut copy in regions {
            copy.clamp_size_to_virtual(&src.copy_size, &dst.copy_size);
            self.cmd_buffer.commands.push(C::CopyTextureToTexture {
                src: src_raw,
                src_target,
                dst: dst_raw,
                dst_target,
                copy,
                dst_is_cubemap: dst.is_cubemap,
            })
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
        let (dst_raw, dst_target) = dst.inner.as_native();

        for mut copy in regions {
            copy.clamp_size_to_virtual(&dst.copy_size);
            self.cmd_buffer.commands.push(C::CopyBufferToTexture {
                src: src.clone(),
                src_target: src.target,
                dst: dst_raw,
                dst_target,
                dst_format: dst.format,
                copy,
            })
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
        let (src_raw, src_target) = src.inner.as_native();
        for mut copy in regions {
            copy.clamp_size_to_virtual(&src.copy_size);
            self.cmd_buffer.commands.push(C::CopyTextureToBuffer {
                src: src_raw,
                src_target,
                src_format: src.format,
                dst: dst.clone(),
                dst_target: dst.target,
                copy,
            })
        }
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        let query = set.queries[index as usize];
        self.cmd_buffer
            .commands
            .push(C::BeginQuery(query, set.target));
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, _index: u32) {
        self.cmd_buffer.commands.push(C::EndQuery(set.target));
    }
    unsafe fn write_timestamp(&mut self, _set: &super::QuerySet, _index: u32) {
        unimplemented!()
    }
    unsafe fn reset_queries(&mut self, _set: &super::QuerySet, _range: Range<u32>) {
        //TODO: what do we do here?
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        _stride: wgt::BufferSize,
    ) {
        let start = self.cmd_buffer.queries.len();
        self.cmd_buffer
            .queries
            .extend_from_slice(&set.queries[range.start as usize..range.end as usize]);
        let query_range = start as u32..self.cmd_buffer.queries.len() as u32;
        self.cmd_buffer.commands.push(C::CopyQueryResults {
            query_range,
            dst: buffer.clone(),
            dst_target: buffer.target,
            dst_offset: offset,
        });
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
        self.state.render_size = desc.extent;
        self.state.resolve_attachments.clear();
        self.state.invalidate_attachments.clear();
        if let Some(label) = desc.label {
            let range = self.cmd_buffer.add_marker(label);
            self.cmd_buffer.commands.push(C::PushDebugGroup(range));
            self.state.has_pass_label = true;
        }

        let rendering_to_external_framebuffer = desc
            .color_attachments
            .iter()
            .filter_map(|at| at.as_ref())
            .any(|at| match at.target.view.inner {
                #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
                super::TextureInner::ExternalFramebuffer { .. } => true,
                _ => false,
            });

        if rendering_to_external_framebuffer && desc.color_attachments.len() != 1 {
            panic!("Multiple render attachments with external framebuffers are not supported.");
        }

        match desc
            .color_attachments
            .first()
            .filter(|at| at.is_some())
            .and_then(|at| at.as_ref().map(|at| &at.target.view.inner))
        {
            // default framebuffer (provided externally)
            Some(&super::TextureInner::DefaultRenderbuffer) => {
                self.cmd_buffer
                    .commands
                    .push(C::ResetFramebuffer { is_default: true });
            }
            _ => {
                // set the framebuffer
                self.cmd_buffer
                    .commands
                    .push(C::ResetFramebuffer { is_default: false });

                for (i, cat) in desc.color_attachments.iter().enumerate() {
                    if let Some(cat) = cat.as_ref() {
                        let attachment = glow::COLOR_ATTACHMENT0 + i as u32;
                        self.cmd_buffer.commands.push(C::BindAttachment {
                            attachment,
                            view: cat.target.view.clone(),
                        });
                        if let Some(ref rat) = cat.resolve_target {
                            self.state
                                .resolve_attachments
                                .push((attachment, rat.view.clone()));
                        }
                        if !cat.ops.contains(crate::AttachmentOps::STORE) {
                            self.state.invalidate_attachments.push(attachment);
                        }
                    }
                }
                if let Some(ref dsat) = desc.depth_stencil_attachment {
                    let aspects = dsat.target.view.aspects;
                    let attachment = match aspects {
                        crate::FormatAspects::DEPTH => glow::DEPTH_ATTACHMENT,
                        crate::FormatAspects::STENCIL => glow::STENCIL_ATTACHMENT,
                        _ => glow::DEPTH_STENCIL_ATTACHMENT,
                    };
                    self.cmd_buffer.commands.push(C::BindAttachment {
                        attachment,
                        view: dsat.target.view.clone(),
                    });
                    if aspects.contains(crate::FormatAspects::DEPTH)
                        && !dsat.depth_ops.contains(crate::AttachmentOps::STORE)
                    {
                        self.state
                            .invalidate_attachments
                            .push(glow::DEPTH_ATTACHMENT);
                    }
                    if aspects.contains(crate::FormatAspects::STENCIL)
                        && !dsat.stencil_ops.contains(crate::AttachmentOps::STORE)
                    {
                        self.state
                            .invalidate_attachments
                            .push(glow::STENCIL_ATTACHMENT);
                    }
                }

                if !rendering_to_external_framebuffer {
                    // set the draw buffers and states
                    self.cmd_buffer
                        .commands
                        .push(C::SetDrawColorBuffers(desc.color_attachments.len() as u8));
                }
            }
        }

        let rect = crate::Rect {
            x: 0,
            y: 0,
            w: desc.extent.width as i32,
            h: desc.extent.height as i32,
        };
        self.cmd_buffer.commands.push(C::SetScissor(rect.clone()));
        self.cmd_buffer.commands.push(C::SetViewport {
            rect,
            depth: 0.0..1.0,
        });

        // issue the clears
        for (i, cat) in desc
            .color_attachments
            .iter()
            .filter_map(|at| at.as_ref())
            .enumerate()
        {
            if !cat.ops.contains(crate::AttachmentOps::LOAD) {
                let c = &cat.clear_value;
                self.cmd_buffer.commands.push(
                    match cat.target.view.format.sample_type(None).unwrap() {
                        wgt::TextureSampleType::Float { .. } => C::ClearColorF {
                            draw_buffer: i as u32,
                            color: [c.r as f32, c.g as f32, c.b as f32, c.a as f32],
                            is_srgb: cat.target.view.format.is_srgb(),
                        },
                        wgt::TextureSampleType::Uint => C::ClearColorU(
                            i as u32,
                            [c.r as u32, c.g as u32, c.b as u32, c.a as u32],
                        ),
                        wgt::TextureSampleType::Sint => C::ClearColorI(
                            i as u32,
                            [c.r as i32, c.g as i32, c.b as i32, c.a as i32],
                        ),
                        wgt::TextureSampleType::Depth => unreachable!(),
                    },
                );
            }
        }
        if let Some(ref dsat) = desc.depth_stencil_attachment {
            let clear_depth = !dsat.depth_ops.contains(crate::AttachmentOps::LOAD);
            let clear_stencil = !dsat.stencil_ops.contains(crate::AttachmentOps::LOAD);

            if clear_depth && clear_stencil {
                self.cmd_buffer.commands.push(C::ClearDepthAndStencil(
                    dsat.clear_value.0,
                    dsat.clear_value.1,
                ));
            } else if clear_depth {
                self.cmd_buffer
                    .commands
                    .push(C::ClearDepth(dsat.clear_value.0));
            } else if clear_stencil {
                self.cmd_buffer
                    .commands
                    .push(C::ClearStencil(dsat.clear_value.1));
            }
        }
    }
    unsafe fn end_render_pass(&mut self) {
        for (attachment, dst) in self.state.resolve_attachments.drain(..) {
            self.cmd_buffer.commands.push(C::ResolveAttachment {
                attachment,
                dst,
                size: self.state.render_size,
            });
        }
        if !self.state.invalidate_attachments.is_empty() {
            self.cmd_buffer.commands.push(C::InvalidateAttachments(
                self.state.invalidate_attachments.clone(),
            ));
            self.state.invalidate_attachments.clear();
        }
        if self.state.has_pass_label {
            self.cmd_buffer.commands.push(C::PopDebugGroup);
            self.state.has_pass_label = false;
        }
        self.state.instance_vbuf_mask = 0;
        self.state.dirty_vbuf_mask = 0;
        self.state.active_first_instance = 0;
        self.state.color_targets.clear();
        for vat in &self.state.vertex_attributes {
            self.cmd_buffer
                .commands
                .push(C::UnsetVertexAttribute(vat.location));
        }
        self.state.vertex_attributes.clear();
        self.state.primitive = super::PrimitiveState::default();
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let mut do_index = 0;
        let mut dirty_textures = 0u32;
        let mut dirty_samplers = 0u32;
        let group_info = &layout.group_infos[index as usize];

        for (binding_layout, raw_binding) in group_info.entries.iter().zip(group.contents.iter()) {
            let slot = group_info.binding_to_slot[binding_layout.binding as usize] as u32;
            match *raw_binding {
                super::RawBinding::Buffer {
                    raw,
                    offset: base_offset,
                    size,
                } => {
                    let mut offset = base_offset;
                    let target = match binding_layout.ty {
                        wgt::BindingType::Buffer {
                            ty,
                            has_dynamic_offset,
                            min_binding_size: _,
                        } => {
                            if has_dynamic_offset {
                                offset += dynamic_offsets[do_index] as i32;
                                do_index += 1;
                            }
                            match ty {
                                wgt::BufferBindingType::Uniform => glow::UNIFORM_BUFFER,
                                wgt::BufferBindingType::Storage { .. } => {
                                    glow::SHADER_STORAGE_BUFFER
                                }
                            }
                        }
                        _ => unreachable!(),
                    };
                    self.cmd_buffer.commands.push(C::BindBuffer {
                        target,
                        slot,
                        buffer: raw,
                        offset,
                        size,
                    });
                }
                super::RawBinding::Sampler(sampler) => {
                    dirty_samplers |= 1 << slot;
                    self.state.samplers[slot as usize] = Some(sampler);
                }
                super::RawBinding::Texture {
                    raw,
                    target,
                    aspects,
                } => {
                    dirty_textures |= 1 << slot;
                    self.state.texture_slots[slot as usize].tex_target = target;
                    self.cmd_buffer.commands.push(C::BindTexture {
                        slot,
                        texture: raw,
                        target,
                        aspects,
                    });
                }
                super::RawBinding::Image(ref binding) => {
                    self.cmd_buffer.commands.push(C::BindImage {
                        slot,
                        binding: binding.clone(),
                    });
                }
            }
        }

        self.rebind_sampler_states(dirty_textures, dirty_samplers);
    }

    unsafe fn set_push_constants(
        &mut self,
        _layout: &super::PipelineLayout,
        _stages: wgt::ShaderStages,
        start_offset: u32,
        data: &[u32],
    ) {
        let range = self.cmd_buffer.add_push_constant_data(data);

        let end = start_offset + data.len() as u32 * 4;
        let mut offset = start_offset;
        while offset < end {
            let uniform = self.state.push_offset_to_uniform[offset as usize / 4].clone();
            let size = uniform.size;
            if uniform.location.is_none() {
                panic!("No uniform for push constant");
            }
            self.cmd_buffer.commands.push(C::SetPushConstants {
                uniform,
                offset: range.start + offset,
            });
            offset += size;
        }
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        let range = self.cmd_buffer.add_marker(label);
        self.cmd_buffer.commands.push(C::InsertDebugMarker(range));
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        let range = self.cmd_buffer.add_marker(group_label);
        self.cmd_buffer.commands.push(C::PushDebugGroup(range));
    }
    unsafe fn end_debug_marker(&mut self) {
        self.cmd_buffer.commands.push(C::PopDebugGroup);
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        self.state.topology = conv::map_primitive_topology(pipeline.primitive.topology);

        if self
            .private_caps
            .contains(super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT)
        {
            for vat in pipeline.vertex_attributes.iter() {
                let vb = &pipeline.vertex_buffers[vat.buffer_index as usize];
                // set the layout
                self.cmd_buffer.commands.push(C::SetVertexAttribute {
                    buffer: None,
                    buffer_desc: vb.clone(),
                    attribute_desc: vat.clone(),
                });
            }
        } else {
            for vat in &self.state.vertex_attributes {
                self.cmd_buffer
                    .commands
                    .push(C::UnsetVertexAttribute(vat.location));
            }
            self.state.vertex_attributes.clear();

            self.state.dirty_vbuf_mask = 0;
            // copy vertex attributes
            for vat in pipeline.vertex_attributes.iter() {
                //Note: we can invalidate more carefully here.
                self.state.dirty_vbuf_mask |= 1 << vat.buffer_index;
                self.state.vertex_attributes.push(vat.clone());
            }
        }

        self.state.instance_vbuf_mask = 0;
        // copy vertex state
        for (index, (&mut (ref mut state_desc, _), pipe_desc)) in self
            .state
            .vertex_buffers
            .iter_mut()
            .zip(pipeline.vertex_buffers.iter())
            .enumerate()
        {
            if pipe_desc.step == wgt::VertexStepMode::Instance {
                self.state.instance_vbuf_mask |= 1 << index;
            }
            if state_desc != pipe_desc {
                self.state.dirty_vbuf_mask |= 1 << index;
                *state_desc = pipe_desc.clone();
            }
        }

        self.set_pipeline_inner(&pipeline.inner);

        // set primitive state
        let prim_state = conv::map_primitive_state(&pipeline.primitive);
        if prim_state != self.state.primitive {
            self.cmd_buffer
                .commands
                .push(C::SetPrimitive(prim_state.clone()));
            self.state.primitive = prim_state;
        }

        // set depth/stencil states
        let mut aspects = crate::FormatAspects::empty();
        if pipeline.depth_bias != self.state.depth_bias {
            self.state.depth_bias = pipeline.depth_bias;
            self.cmd_buffer
                .commands
                .push(C::SetDepthBias(pipeline.depth_bias));
        }
        if let Some(ref depth) = pipeline.depth {
            aspects |= crate::FormatAspects::DEPTH;
            self.cmd_buffer.commands.push(C::SetDepth(depth.clone()));
        }
        if let Some(ref stencil) = pipeline.stencil {
            aspects |= crate::FormatAspects::STENCIL;
            self.state.stencil = stencil.clone();
            self.rebind_stencil_func();
            if stencil.front.ops == stencil.back.ops
                && stencil.front.mask_write == stencil.back.mask_write
            {
                self.cmd_buffer.commands.push(C::SetStencilOps {
                    face: glow::FRONT_AND_BACK,
                    write_mask: stencil.front.mask_write,
                    ops: stencil.front.ops.clone(),
                });
            } else {
                self.cmd_buffer.commands.push(C::SetStencilOps {
                    face: glow::FRONT,
                    write_mask: stencil.front.mask_write,
                    ops: stencil.front.ops.clone(),
                });
                self.cmd_buffer.commands.push(C::SetStencilOps {
                    face: glow::BACK,
                    write_mask: stencil.back.mask_write,
                    ops: stencil.back.ops.clone(),
                });
            }
        }
        self.cmd_buffer
            .commands
            .push(C::ConfigureDepthStencil(aspects));

        // set multisampling state
        if pipeline.alpha_to_coverage_enabled != self.state.alpha_to_coverage_enabled {
            self.state.alpha_to_coverage_enabled = pipeline.alpha_to_coverage_enabled;
            self.cmd_buffer
                .commands
                .push(C::SetAlphaToCoverage(pipeline.alpha_to_coverage_enabled));
        }

        // set blend states
        if self.state.color_targets[..] != pipeline.color_targets[..] {
            if pipeline
                .color_targets
                .iter()
                .skip(1)
                .any(|ct| *ct != pipeline.color_targets[0])
            {
                for (index, ct) in pipeline.color_targets.iter().enumerate() {
                    self.cmd_buffer.commands.push(C::SetColorTarget {
                        draw_buffer_index: Some(index as u32),
                        desc: ct.clone(),
                    });
                }
            } else {
                self.cmd_buffer.commands.push(C::SetColorTarget {
                    draw_buffer_index: None,
                    desc: pipeline.color_targets.first().cloned().unwrap_or_default(),
                });
            }
        }
        self.state.color_targets.clear();
        for ct in pipeline.color_targets.iter() {
            self.state.color_targets.push(ct.clone());
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
        self.state.index_offset = binding.offset;
        self.state.index_format = format;
        self.cmd_buffer
            .commands
            .push(C::SetIndexBuffer(binding.buffer.raw.unwrap()));
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
        self.state.dirty_vbuf_mask |= 1 << index;
        let (_, ref mut vb) = self.state.vertex_buffers[index as usize];
        *vb = Some(super::BufferBinding {
            raw: binding.buffer.raw.unwrap(),
            offset: binding.offset,
        });
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth: Range<f32>) {
        self.cmd_buffer.commands.push(C::SetViewport {
            rect: crate::Rect {
                x: rect.x as i32,
                y: rect.y as i32,
                w: rect.w as i32,
                h: rect.h as i32,
            },
            depth,
        });
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        self.cmd_buffer.commands.push(C::SetScissor(crate::Rect {
            x: rect.x as i32,
            y: rect.y as i32,
            w: rect.w as i32,
            h: rect.h as i32,
        }));
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.state.stencil.front.reference = value;
        self.state.stencil.back.reference = value;
        self.rebind_stencil_func();
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        self.cmd_buffer.commands.push(C::SetBlendConstant(*color));
    }

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.prepare_draw(start_instance);
        self.cmd_buffer.commands.push(C::Draw {
            topology: self.state.topology,
            start_vertex,
            vertex_count,
            instance_count,
        });
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.prepare_draw(start_instance);
        let (index_size, index_type) = match self.state.index_format {
            wgt::IndexFormat::Uint16 => (2, glow::UNSIGNED_SHORT),
            wgt::IndexFormat::Uint32 => (4, glow::UNSIGNED_INT),
        };
        let index_offset = self.state.index_offset + index_size * start_index as wgt::BufferAddress;
        self.cmd_buffer.commands.push(C::DrawIndexed {
            topology: self.state.topology,
            index_type,
            index_offset,
            index_count,
            base_vertex,
            instance_count,
        });
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.prepare_draw(0);
        for draw in 0..draw_count as wgt::BufferAddress {
            let indirect_offset =
                offset + draw * mem::size_of::<wgt::DrawIndirectArgs>() as wgt::BufferAddress;
            self.cmd_buffer.commands.push(C::DrawIndirect {
                topology: self.state.topology,
                indirect_buf: buffer.raw.unwrap(),
                indirect_offset,
            });
        }
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.prepare_draw(0);
        let index_type = match self.state.index_format {
            wgt::IndexFormat::Uint16 => glow::UNSIGNED_SHORT,
            wgt::IndexFormat::Uint32 => glow::UNSIGNED_INT,
        };
        for draw in 0..draw_count as wgt::BufferAddress {
            let indirect_offset = offset
                + draw * mem::size_of::<wgt::DrawIndexedIndirectArgs>() as wgt::BufferAddress;
            self.cmd_buffer.commands.push(C::DrawIndexedIndirect {
                topology: self.state.topology,
                index_type,
                indirect_buf: buffer.raw.unwrap(),
                indirect_offset,
            });
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
        unreachable!()
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        _buffer: &super::Buffer,
        _offset: wgt::BufferAddress,
        _count_buffer: &super::Buffer,
        _count_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        unreachable!()
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {
        if let Some(label) = desc.label {
            let range = self.cmd_buffer.add_marker(label);
            self.cmd_buffer.commands.push(C::PushDebugGroup(range));
            self.state.has_pass_label = true;
        }
    }
    unsafe fn end_compute_pass(&mut self) {
        if self.state.has_pass_label {
            self.cmd_buffer.commands.push(C::PopDebugGroup);
            self.state.has_pass_label = false;
        }
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {
        self.set_pipeline_inner(&pipeline.inner);
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        self.cmd_buffer.commands.push(C::Dispatch(count));
    }
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.cmd_buffer.commands.push(C::DispatchIndirect {
            indirect_buf: buffer.raw.unwrap(),
            indirect_offset: offset,
        });
    }
}
