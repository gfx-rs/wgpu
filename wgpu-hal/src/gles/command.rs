use super::{conv, Command as C};
use std::{mem, ops::Range};

impl super::CommandBuffer {
    fn clear(&mut self) {
        self.label = None;
        self.commands.clear();
        self.data_bytes.clear();
        self.data_words.clear();
    }

    fn add_marker(&mut self, marker: &str) -> Range<u32> {
        let start = self.data_bytes.len() as u32;
        self.data_bytes.extend(marker.as_bytes());
        start..self.data_bytes.len() as u32
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        self.state = super::CommandState::default();
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
            .contains(super::PrivateCapability::MEMORY_BARRIERS)
        {
            return;
        }
        for bar in barriers {
            // GLES only synchronizes storage -> anything explicitly
            if !bar.usage.start.contains(crate::BufferUse::STORAGE_STORE) {
                continue;
            }
            self.cmd_buffer
                .commands
                .push(C::BufferBarrier(bar.buffer.raw, bar.usage.end));
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        if !self
            .private_caps
            .contains(super::PrivateCapability::MEMORY_BARRIERS)
        {
            return;
        }
        for bar in barriers {
            // GLES only synchronizes storage -> anything explicitly
            if !bar.usage.start.contains(crate::TextureUse::STORAGE_STORE) {
                continue;
            }
            let raw = match bar.texture.inner {
                super::TextureInner::Texture { raw, .. } => raw,
                super::TextureInner::Renderbuffer { .. } => continue,
            };
            self.cmd_buffer
                .commands
                .push(C::TextureBarrier(raw, bar.usage.end));
        }
    }

    unsafe fn fill_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange, value: u8) {
        self.cmd_buffer.commands.push(C::FillBuffer {
            dst: buffer.raw,
            range,
            value,
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
        for copy in regions {
            self.cmd_buffer.commands.push(C::CopyBufferToBuffer {
                src: src.raw,
                src_target: src.target,
                dst: dst.raw,
                dst_target: dst.target,
                copy,
            })
        }
    }

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUse,
        dst: &super::Texture,
        regions: T,
    ) where
        T: Iterator<Item = crate::TextureCopy>,
    {
        let (src_raw, src_target) = src.inner.as_native();
        let (dst_raw, dst_target) = dst.inner.as_native();
        for copy in regions {
            self.cmd_buffer.commands.push(C::CopyTextureToTexture {
                src: src_raw,
                src_target,
                dst: dst_raw,
                dst_target,
                copy,
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
        let format_info = dst.format.describe();
        assert_eq!(
            format_info.block_dimensions,
            (1, 1),
            "Compressed texture copies are TODO"
        );
        let (dst_raw, dst_target) = dst.inner.as_native();
        for copy in regions {
            self.cmd_buffer.commands.push(C::CopyBufferToTexture {
                src: src.raw,
                src_target: src.target,
                dst: dst_raw,
                dst_target,
                dst_info: super::TextureCopyInfo {
                    external_format: dst.format_desc.external,
                    data_type: dst.format_desc.data_type,
                    texel_size: format_info.block_size,
                },
                copy,
            })
        }
    }

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &super::Texture,
        _src_usage: crate::TextureUse,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferTextureCopy>,
    {
        let format_info = src.format.describe();
        assert_eq!(
            format_info.block_dimensions,
            (1, 1),
            "Compressed texture copies are TODO"
        );
        let (src_raw, src_target) = src.inner.as_native();
        for copy in regions {
            self.cmd_buffer.commands.push(C::CopyTextureToBuffer {
                src: src_raw,
                src_target,
                src_info: super::TextureCopyInfo {
                    external_format: src.format_desc.external,
                    data_type: src.format_desc.data_type,
                    texel_size: format_info.block_size,
                },
                dst: dst.raw,
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
    ) {
        let start = self.cmd_buffer.data_words.len();
        self.cmd_buffer
            .data_words
            .extend_from_slice(&set.queries[range.start as usize..range.end as usize]);
        let query_range = start as u32..self.cmd_buffer.data_words.len() as u32;
        self.cmd_buffer.commands.push(C::CopyQueryResults {
            query_range,
            dst: buffer.raw,
            dst_target: buffer.target,
            dst_offset: offset,
        });
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
        if let Some(label) = desc.label {
            let range = self.cmd_buffer.add_marker(label);
            self.cmd_buffer.commands.push(C::PushDebugGroup(range));
            self.state.has_pass_label = true;
        }

        // set the framebuffer
        self.cmd_buffer
            .commands
            .push(C::ResetFramebuffer(desc.extent));
        for (i, cat) in desc.color_attachments.iter().enumerate() {
            let attachment = glow::COLOR_ATTACHMENT0 + i as u32;
            self.cmd_buffer.commands.push(C::SetFramebufferAttachment {
                attachment,
                view: cat.target.view.clone(),
            });
        }
        if let Some(ref dsat) = desc.depth_stencil_attachment {
            let attachment = match dsat.target.view.aspects {
                crate::FormatAspect::DEPTH => glow::DEPTH_ATTACHMENT,
                crate::FormatAspect::STENCIL => glow::STENCIL_ATTACHMENT,
                _ => glow::DEPTH_STENCIL_ATTACHMENT,
            };
            self.cmd_buffer.commands.push(C::SetFramebufferAttachment {
                attachment,
                view: dsat.target.view.clone(),
            });
        }

        self.cmd_buffer
            .commands
            .push(C::SetDrawColorBuffers(desc.color_attachments.len() as u8));
        // issue the clears
        for (i, cat) in desc.color_attachments.iter().enumerate() {
            if !cat.ops.contains(crate::AttachmentOp::LOAD) {
                let draw_buffer = glow::DRAW_BUFFER0 + i as u32;
                let c = &cat.clear_value;
                self.cmd_buffer
                    .commands
                    .push(match cat.target.view.sample_type {
                        wgt::TextureSampleType::Float { .. } => C::ClearColorF(
                            draw_buffer,
                            [c.r as f32, c.g as f32, c.r as f32, c.a as f32],
                        ),
                        wgt::TextureSampleType::Depth => unimplemented!(),
                        wgt::TextureSampleType::Uint => C::ClearColorU(
                            draw_buffer,
                            [c.r as u32, c.g as u32, c.r as u32, c.a as u32],
                        ),
                        wgt::TextureSampleType::Sint => C::ClearColorI(
                            draw_buffer,
                            [c.r as i32, c.g as i32, c.r as i32, c.a as i32],
                        ),
                    });
            }
        }
        if let Some(ref dsat) = desc.depth_stencil_attachment {
            if !dsat.depth_ops.contains(crate::AttachmentOp::LOAD) {
                self.cmd_buffer
                    .commands
                    .push(C::ClearDepth(dsat.clear_value.0));
            }
            if !dsat.stencil_ops.contains(crate::AttachmentOp::LOAD) {
                self.cmd_buffer
                    .commands
                    .push(C::ClearStencil(dsat.clear_value.1));
            }
        }
    }
    unsafe fn end_render_pass(&mut self) {
        if self.state.has_pass_label {
            self.cmd_buffer.commands.push(C::PopDebugGroup);
            self.state.has_pass_label = false;
        }
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
    }
    unsafe fn set_push_constants(
        &mut self,
        layout: &super::PipelineLayout,
        stages: wgt::ShaderStage,
        offset: u32,
        data: &[u32],
    ) {
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
            .push(C::SetIndexBuffer(binding.buffer.raw));
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {}
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {}
    unsafe fn set_stencil_reference(&mut self, value: u32) {}
    unsafe fn set_blend_constants(&mut self, color: &wgt::Color) {}

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        debug_assert_eq!(start_instance, 0);
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
        debug_assert_eq!(start_instance, 0);
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
        for draw in 0..draw_count as wgt::BufferAddress {
            let indirect_offset =
                offset + draw * mem::size_of::<wgt::DrawIndirectArgs>() as wgt::BufferAddress;
            self.cmd_buffer.commands.push(C::DrawIndirect {
                topology: self.state.topology,
                indirect_buf: buffer.raw,
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
                indirect_buf: buffer.raw,
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
        unimplemented!()
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        _buffer: &super::Buffer,
        _offset: wgt::BufferAddress,
        _count_buffer: &super::Buffer,
        _count_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        unimplemented!()
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

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        self.cmd_buffer.commands.push(C::Dispatch(count));
    }
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.cmd_buffer.commands.push(C::DispatchIndirect {
            indirect_buf: buffer.raw,
            indirect_offset: offset,
        });
    }
}
