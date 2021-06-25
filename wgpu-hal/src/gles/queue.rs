use super::Command as C;
use arrayvec::ArrayVec;
use glow::HasContext;
use std::{mem, ops::Range, slice};

const DEBUG_ID: u32 = 0;

fn extract_marker<'a>(data: &'a [u8], range: &Range<u32>) -> &'a str {
    std::str::from_utf8(&data[range.start as usize..range.end as usize]).unwrap()
}

fn is_3d_target(target: super::BindTarget) -> bool {
    match target {
        glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => true,
        _ => false,
    }
}

impl super::Queue {
    unsafe fn reset_state(&self) {
        let gl = &self.shared.context;
        gl.use_program(None);
        gl.polygon_offset(0.0, 0.0);
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::STENCIL_TEST);
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::BLEND);
    }

    unsafe fn process(&mut self, command: &C, data_bytes: &[u8], data_words: &[u32]) {
        let gl = &self.shared.context;
        match *command {
            C::Draw {
                topology,
                start_vertex,
                vertex_count,
                instance_count,
            } => {
                if instance_count == 1 {
                    gl.draw_arrays(topology, start_vertex as i32, vertex_count as i32);
                } else {
                    gl.draw_arrays_instanced(
                        topology,
                        start_vertex as i32,
                        vertex_count as i32,
                        instance_count as i32,
                    );
                }
            }
            C::DrawIndexed {
                topology,
                index_type,
                index_count,
                index_offset,
                base_vertex,
                instance_count,
            } => match (base_vertex, instance_count) {
                (0, 1) => gl.draw_elements(
                    topology,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                ),
                (0, _) => gl.draw_elements_instanced(
                    topology,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                    instance_count as i32,
                ),
                (_, 1) => gl.draw_elements_base_vertex(
                    topology,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                    base_vertex,
                ),
                (_, _) => gl.draw_elements_instanced_base_vertex(
                    topology,
                    index_count as _,
                    index_type,
                    index_offset as i32,
                    instance_count as i32,
                    base_vertex,
                ),
            },
            C::DrawIndirect {
                topology: _,
                indirect_buf,
                indirect_offset: _,
            } => {
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf));
                //TODO: https://github.com/grovesNL/glow/issues/172
                //gl.draw_arrays_indirect(topology, indirect_offset);
            }
            C::DrawIndexedIndirect {
                topology: _,
                index_type: _,
                indirect_buf,
                indirect_offset: _,
            } => {
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf));
                //TODO: https://github.com/grovesNL/glow/issues/172
            }
            C::Dispatch(group_counts) => {
                gl.dispatch_compute(group_counts[0], group_counts[1], group_counts[2]);
            }
            C::DispatchIndirect {
                indirect_buf,
                indirect_offset,
            } => {
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf));
                gl.dispatch_compute_indirect(indirect_offset as i32);
            }
            C::FillBuffer { .. } => unimplemented!(),
            C::CopyBufferToBuffer {
                src,
                src_target,
                dst,
                dst_target,
                copy,
            } => {
                gl.bind_buffer(src_target, Some(src));
                gl.bind_buffer(dst_target, Some(dst));

                gl.copy_buffer_sub_data(
                    src_target,
                    dst_target,
                    copy.src_offset as i32,
                    copy.dst_offset as i32,
                    copy.size.get() as i32,
                );
            }
            C::CopyTextureToTexture {
                src,
                src_target,
                dst,
                dst_target,
                ref copy,
            } => {
                //TODO: how is depth handled?
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.copy_fbo));
                for layer in 0..copy.size.depth_or_array_layers as i32 {
                    if is_3d_target(src_target) {
                        //TODO: handle GLES without framebuffer_texture_3d
                        gl.framebuffer_texture_layer(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            Some(src),
                            copy.src_base.mip_level as i32,
                            copy.src_base.origin.z as i32 + layer,
                        );
                    } else {
                        gl.framebuffer_texture_2d(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            src_target,
                            Some(src),
                            copy.src_base.mip_level as i32,
                        );
                    }
                    gl.bind_texture(dst_target, Some(dst));
                    //TODO: https://github.com/grovesNL/glow/issues/173
                    #[allow(clippy::if_same_then_else)]
                    if is_3d_target(dst_target) {
                        //gl.copy_tex_sub_image_3d(dst_target, copy.dst_base.mip_level, copy.dst_base.origin.x, copy.dst_base.origin.y, copy.dst_base.origin.z + layer, copy.src_base.origin.x, copy.src_base.origin.y, copy.size.width, copy.size.height);
                    } else {
                        //gl.copy_tex_sub_image_2d(dst_target, copy.dst_base.mip_level, copy.dst_base.origin.x, copy.dst_base.origin.y, copy.src_base.origin.x, copy.src_base.origin.y, copy.size.width, copy.size.height);
                    }
                }
            }
            C::CopyBufferToTexture {
                src,
                src_target: _,
                dst,
                dst_target,
                ref dst_info,
                ref copy,
            } => {
                //TODO: compressed data
                let row_texels = copy
                    .buffer_layout
                    .bytes_per_row
                    .map_or(0, |bpr| bpr.get() / dst_info.texel_size as u32);
                let column_texels = copy.buffer_layout.rows_per_image.map_or(0, |rpi| rpi.get());
                gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, row_texels as i32);
                gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, column_texels as i32);
                gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(src));

                let unpack_data =
                    glow::PixelUnpackData::BufferOffset(copy.buffer_layout.offset as u32);
                gl.bind_texture(dst_target, Some(dst));
                if is_3d_target(dst_target) {
                    gl.tex_sub_image_3d(
                        dst_target,
                        copy.texture_base.mip_level as i32,
                        copy.texture_base.origin.x as i32,
                        copy.texture_base.origin.y as i32,
                        copy.texture_base.origin.z as i32,
                        copy.size.width as i32,
                        copy.size.height as i32,
                        copy.size.depth_or_array_layers as i32,
                        dst_info.external_format,
                        dst_info.data_type,
                        unpack_data,
                    );
                } else {
                    gl.tex_sub_image_2d(
                        dst_target,
                        copy.texture_base.mip_level as i32,
                        copy.texture_base.origin.x as i32,
                        copy.texture_base.origin.y as i32,
                        copy.size.width as i32,
                        copy.size.height as i32,
                        dst_info.external_format,
                        dst_info.data_type,
                        unpack_data,
                    );
                }
            }
            C::CopyTextureToBuffer {
                src,
                src_target,
                ref src_info,
                dst,
                dst_target: _,
                ref copy,
            } => {
                //TODO: compressed data
                let row_texels = copy
                    .buffer_layout
                    .bytes_per_row
                    .map_or(copy.size.width, |bpr| {
                        bpr.get() / src_info.texel_size as u32
                    });
                let column_texels = copy
                    .buffer_layout
                    .rows_per_image
                    .map_or(copy.size.height, |rpi| rpi.get());
                gl.pixel_store_i32(glow::PACK_ROW_LENGTH, row_texels as i32);
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(dst));

                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.copy_fbo));
                for layer in 0..copy.size.depth_or_array_layers {
                    let offset = copy.buffer_layout.offset as u32
                        + layer * column_texels * row_texels * src_info.texel_size as u32;
                    if is_3d_target(src_target) {
                        //TODO: handle GLES without framebuffer_texture_3d
                        gl.framebuffer_texture_layer(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            Some(src),
                            copy.texture_base.mip_level as i32,
                            copy.texture_base.origin.z as i32 + layer as i32,
                        );
                    } else {
                        gl.framebuffer_texture_2d(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            src_target,
                            Some(src),
                            copy.texture_base.mip_level as i32,
                        );
                    }
                    gl.read_pixels(
                        copy.texture_base.origin.x as i32,
                        copy.texture_base.origin.y as i32,
                        copy.size.width as i32,
                        copy.size.height as i32,
                        src_info.external_format,
                        src_info.data_type,
                        glow::PixelPackData::BufferOffset(offset),
                    );
                }
            }
            C::SetIndexBuffer(buffer) => {
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(buffer));
            }
            C::BeginQuery(query, target) => {
                gl.begin_query(target, query);
            }
            C::EndQuery(target) => {
                gl.end_query(target);
            }
            C::CopyQueryResults {
                ref query_range,
                dst,
                dst_target,
                dst_offset,
            } => {
                self.temp_query_results.clear();
                for &query in
                    data_words[query_range.start as usize..query_range.end as usize].iter()
                {
                    let result = gl.get_query_parameter_u32(query, glow::QUERY_RESULT);
                    self.temp_query_results.push(result as u64);
                }
                let query_data = slice::from_raw_parts(
                    self.temp_query_results.as_ptr() as *const u8,
                    self.temp_query_results.len() * mem::size_of::<u64>(),
                );
                gl.bind_buffer(dst_target, Some(dst));
                gl.buffer_sub_data_u8_slice(dst_target, dst_offset as i32, query_data);
            }
            C::ResetFramebuffer => {
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.draw_fbo));
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    glow::DEPTH_STENCIL_ATTACHMENT,
                    glow::TEXTURE_2D,
                    None,
                    0,
                );
                for i in 0..crate::MAX_COLOR_TARGETS {
                    let target = glow::COLOR_ATTACHMENT0 + i as u32;
                    gl.framebuffer_texture_2d(
                        glow::DRAW_FRAMEBUFFER,
                        target,
                        glow::TEXTURE_2D,
                        None,
                        0,
                    );
                }
                gl.color_mask(true, true, true, true);
                gl.depth_mask(true);
                gl.stencil_mask(!0);
                gl.disable(glow::DEPTH_TEST);
                gl.disable(glow::STENCIL_TEST);
                gl.disable(glow::SCISSOR_TEST);
            }
            C::SetFramebufferAttachment {
                attachment,
                ref view,
            } => match view.inner {
                super::TextureInner::Renderbuffer { raw } => {
                    gl.framebuffer_renderbuffer(
                        glow::DRAW_FRAMEBUFFER,
                        attachment,
                        glow::RENDERBUFFER,
                        Some(raw),
                    );
                }
                super::TextureInner::Texture { raw, target } => {
                    if is_3d_target(target) {
                        gl.framebuffer_texture_layer(
                            glow::DRAW_FRAMEBUFFER,
                            attachment,
                            Some(raw),
                            view.mip_levels.start as i32,
                            view.array_layers.start as i32,
                        );
                    } else {
                        gl.framebuffer_texture_2d(
                            glow::DRAW_FRAMEBUFFER,
                            attachment,
                            target,
                            Some(raw),
                            view.mip_levels.start as i32,
                        );
                    }
                }
            },
            C::SetDrawColorBuffers(count) => {
                let indices = (0..count as u32)
                    .map(|i| glow::COLOR_ATTACHMENT0 + i)
                    .collect::<ArrayVec<[_; crate::MAX_COLOR_TARGETS]>>();
                gl.draw_buffers(&indices);
                for draw_buffer in 0..count as u32 {
                    gl.disable_draw_buffer(glow::BLEND, draw_buffer);
                }
            }
            C::ClearColorF(draw_buffer, mut color) => {
                gl.clear_buffer_f32_slice(glow::COLOR, draw_buffer, &mut color);
            }
            C::ClearColorU(draw_buffer, mut color) => {
                gl.clear_buffer_u32_slice(glow::COLOR, draw_buffer, &mut color);
            }
            C::ClearColorI(draw_buffer, mut color) => {
                gl.clear_buffer_i32_slice(glow::COLOR, draw_buffer, &mut color);
            }
            C::ClearDepth(depth) => {
                gl.clear_buffer_depth_stencil(glow::DEPTH, 0, depth, 0);
            }
            C::ClearStencil(value) => {
                gl.clear_buffer_depth_stencil(glow::STENCIL, 0, 0.0, value as i32);
            }
            C::BufferBarrier(raw, usage) => {
                let mut flags = 0;
                if usage.contains(crate::BufferUse::VERTEX) {
                    flags |= glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(raw));
                    gl.vertex_attrib_pointer_f32(0, 1, glow::BYTE, true, 0, 0);
                }
                if usage.contains(crate::BufferUse::INDEX) {
                    flags |= glow::ELEMENT_ARRAY_BARRIER_BIT;
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(raw));
                }
                if usage.contains(crate::BufferUse::UNIFORM) {
                    flags |= glow::UNIFORM_BARRIER_BIT;
                }
                if usage.contains(crate::BufferUse::INDIRECT) {
                    flags |= glow::COMMAND_BARRIER_BIT;
                    gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(raw));
                }
                if usage.contains(crate::BufferUse::COPY_SRC) {
                    flags |= glow::PIXEL_BUFFER_BARRIER_BIT;
                    gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(raw));
                }
                if usage.contains(crate::BufferUse::COPY_DST) {
                    flags |= glow::PIXEL_BUFFER_BARRIER_BIT;
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(raw));
                }
                if usage.intersects(crate::BufferUse::MAP_READ | crate::BufferUse::MAP_WRITE) {
                    flags |= glow::BUFFER_UPDATE_BARRIER_BIT;
                }
                if usage
                    .intersects(crate::BufferUse::STORAGE_LOAD | crate::BufferUse::STORAGE_STORE)
                {
                    flags |= glow::SHADER_STORAGE_BARRIER_BIT;
                }
                gl.memory_barrier(flags);
            }
            C::TextureBarrier(usage) => {
                let mut flags = 0;
                if usage.contains(crate::TextureUse::SAMPLED) {
                    flags |= glow::TEXTURE_FETCH_BARRIER_BIT;
                }
                if usage
                    .intersects(crate::TextureUse::STORAGE_LOAD | crate::TextureUse::STORAGE_STORE)
                {
                    flags |= glow::SHADER_IMAGE_ACCESS_BARRIER_BIT;
                }
                if usage.contains(crate::TextureUse::COPY_DST) {
                    flags |= glow::TEXTURE_UPDATE_BARRIER_BIT;
                }
                if usage.intersects(
                    crate::TextureUse::COLOR_TARGET
                        | crate::TextureUse::DEPTH_STENCIL_READ
                        | crate::TextureUse::DEPTH_STENCIL_WRITE,
                ) {
                    flags |= glow::FRAMEBUFFER_BARRIER_BIT;
                }
                gl.memory_barrier(flags);
            }
            C::SetViewport {
                ref rect,
                ref depth,
            } => {
                gl.viewport(rect.x, rect.y, rect.w, rect.h);
                gl.depth_range_f32(depth.start, depth.end);
            }
            C::SetScissor(ref rect) => {
                gl.scissor(rect.x, rect.y, rect.w, rect.h);
                gl.enable(glow::SCISSOR_TEST);
            }
            C::SetStencilFunc {
                face,
                function,
                reference,
                read_mask,
            } => {
                gl.stencil_func_separate(face, function, reference as i32, read_mask);
            }
            C::SetStencilOps {
                face,
                write_mask,
                ref ops,
            } => {
                gl.stencil_mask_separate(face, write_mask);
                gl.stencil_op_separate(face, ops.fail, ops.depth_fail, ops.pass);
            }
            C::SetVertexAttribute {
                ref buffer_desc,
                ref buffer,
                attribute_desc: ref vat,
            } => {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer.raw));
                let offset = vat.offset as i32 + buffer.offset as i32;
                match vat.format_desc.attrib_kind {
                    super::VertexAttribKind::Float => gl.vertex_attrib_pointer_f32(
                        vat.location,
                        vat.format_desc.element_count,
                        vat.format_desc.element_format,
                        true, // always normalized
                        buffer_desc.stride as i32,
                        offset,
                    ),
                    super::VertexAttribKind::Integer => gl.vertex_attrib_pointer_i32(
                        vat.location,
                        vat.format_desc.element_count,
                        vat.format_desc.element_format,
                        buffer_desc.stride as i32,
                        offset,
                    ),
                }
                gl.vertex_attrib_divisor(vat.location, buffer_desc.step as u32);
                gl.enable_vertex_attrib_array(vat.location);
            }
            C::SetDepth(ref depth) => {
                gl.depth_func(depth.function);
                gl.depth_mask(depth.mask);
            }
            C::SetDepthBias(bias) => {
                gl.polygon_offset(bias.constant as f32, bias.slope_scale);
            }
            C::ConfigureDepthStencil(aspects) => {
                if aspects.contains(crate::FormatAspect::DEPTH) {
                    gl.enable(glow::DEPTH_TEST);
                } else {
                    gl.disable(glow::DEPTH_TEST);
                }
                if aspects.contains(crate::FormatAspect::STENCIL) {
                    gl.enable(glow::STENCIL_TEST);
                } else {
                    gl.disable(glow::STENCIL_TEST);
                }
            }
            C::SetProgram(program) => {
                gl.use_program(Some(program));
            }
            C::SetBlendConstant(c) => {
                gl.blend_color(c[0], c[1], c[2], c[3]);
            }
            C::SetColorTarget {
                draw_buffer_index,
                desc: super::ColorTargetDesc { mask, ref blend },
            } => {
                use wgt::ColorWrite as Cw;
                if let Some(index) = draw_buffer_index {
                    gl.color_mask_draw_buffer(
                        index,
                        mask.contains(Cw::RED),
                        mask.contains(Cw::GREEN),
                        mask.contains(Cw::BLUE),
                        mask.contains(Cw::ALPHA),
                    );
                    if let Some(ref blend) = *blend {
                        gl.enable_draw_buffer(index, glow::BLEND);
                        if blend.color != blend.alpha {
                            gl.blend_equation_separate_draw_buffer(
                                index,
                                blend.color.equation,
                                blend.alpha.equation,
                            );
                            gl.blend_func_separate_draw_buffer(
                                index,
                                blend.color.src,
                                blend.color.dst,
                                blend.alpha.src,
                                blend.alpha.dst,
                            );
                        } else {
                            gl.blend_equation_draw_buffer(index, blend.color.equation);
                            gl.blend_func_draw_buffer(index, blend.color.src, blend.color.dst);
                        }
                    } else {
                        gl.disable_draw_buffer(index, glow::BLEND);
                    }
                } else {
                    gl.color_mask(
                        mask.contains(Cw::RED),
                        mask.contains(Cw::GREEN),
                        mask.contains(Cw::BLUE),
                        mask.contains(Cw::ALPHA),
                    );
                    if let Some(ref blend) = *blend {
                        gl.enable(glow::BLEND);
                        if blend.color != blend.alpha {
                            gl.blend_equation_separate(blend.color.equation, blend.alpha.equation);
                            gl.blend_func_separate(
                                blend.color.src,
                                blend.color.dst,
                                blend.alpha.src,
                                blend.alpha.dst,
                            );
                        } else {
                            gl.blend_equation(blend.color.equation);
                            gl.blend_func(blend.color.src, blend.color.dst);
                        }
                    } else {
                        gl.disable(glow::BLEND);
                    }
                }
            }
            C::BindBuffer {
                target,
                slot,
                buffer,
                offset,
                size,
            } => {
                gl.bind_buffer_range(target, slot, Some(buffer), offset, size);
            }
            C::BindSampler(texture_index, sampler) => {
                gl.bind_sampler(texture_index, Some(sampler));
            }
            C::BindTexture {
                slot,
                texture,
                target,
            } => {
                gl.active_texture(glow::TEXTURE0 + slot);
                gl.bind_texture(target, Some(texture));
            }
            C::BindImage {
                slot: _,
                binding: _,
            } => {
                //TODO: https://github.com/grovesNL/glow/issues/174
                //gl.bind_image_texture(slot, Some(binding.raw), binding.mip_level as i32,
                //    binding.array_layer.is_none(), binding.array_layer.unwrap_or_default(),
                //    binding.access, binding.format);
            }
            C::InsertDebugMarker(ref range) => {
                let marker = extract_marker(data_bytes, range);
                gl.debug_message_insert(
                    glow::DEBUG_SOURCE_APPLICATION,
                    glow::DEBUG_TYPE_MARKER,
                    DEBUG_ID,
                    glow::DEBUG_SEVERITY_NOTIFICATION,
                    marker,
                );
            }
            C::PushDebugGroup(ref range) => {
                let marker = extract_marker(data_bytes, range);
                gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, DEBUG_ID, marker);
            }
            C::PopDebugGroup => {
                gl.pop_debug_group();
            }
        }
    }
}

impl crate::Queue<super::Api> for super::Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&super::CommandBuffer],
        signal_fence: Option<(&mut super::Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        self.reset_state();
        for cmd_buf in command_buffers.iter() {
            for command in cmd_buf.commands.iter() {
                self.process(command, &cmd_buf.data_bytes, &cmd_buf.data_words);
            }
        }

        if let Some((fence, value)) = signal_fence {
            let gl = &self.shared.context;
            fence.maintain(gl);
            let sync = gl
                .fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)
                .map_err(|_| crate::DeviceError::OutOfMemory)?;
            fence.pending.push((value, sync));
        }

        Ok(())
    }

    unsafe fn present(
        &mut self,
        surface: &mut super::Surface,
        texture: super::Texture,
    ) -> Result<(), crate::SurfaceError> {
        let gl = &self.shared.context;
        surface.present(texture, gl)
    }
}
