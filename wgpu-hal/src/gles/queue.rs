use super::Command as C;
use glow::HasContext;
use std::ops::Range;

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
    unsafe fn process(&mut self, command: &C, data: &[u8]) {
        let gl = &self.shared.context;
        match *command {
            C::Draw {
                primitive,
                start_vertex,
                vertex_count,
                instance_count,
            } => {
                if instance_count == 1 {
                    gl.draw_arrays(primitive, start_vertex as i32, vertex_count as i32);
                } else {
                    gl.draw_arrays_instanced(
                        primitive,
                        start_vertex as i32,
                        vertex_count as i32,
                        instance_count as i32,
                    );
                }
            }
            C::DrawIndexed {
                primitive,
                index_type,
                index_count,
                index_offset,
                base_vertex,
                instance_count,
            } => match (base_vertex, instance_count) {
                (0, 1) => gl.draw_elements(
                    primitive,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                ),
                (0, _) => gl.draw_elements_instanced(
                    primitive,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                    instance_count as i32,
                ),
                (_, 1) => gl.draw_elements_base_vertex(
                    primitive,
                    index_count as i32,
                    index_type,
                    index_offset as i32,
                    base_vertex,
                ),
                (_, _) => gl.draw_elements_instanced_base_vertex(
                    primitive,
                    index_count as _,
                    index_type,
                    index_offset as i32,
                    instance_count as i32,
                    base_vertex,
                ),
            },
            C::DrawIndirect {
                primitive: _,
                indirect_buf,
                indirect_offset: _,
            } => {
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf));
                //TODO: https://github.com/grovesNL/glow/issues/172
                //gl.draw_arrays_indirect(primitive, indirect_offset);
            }
            C::DrawIndexedIndirect {
                primitive: _,
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
                src_target,
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
                dst_target,
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
            C::InsertDebugMarker(ref range) => {
                let marker = extract_marker(data, range);
                gl.debug_message_insert(
                    glow::DEBUG_SOURCE_APPLICATION,
                    glow::DEBUG_TYPE_MARKER,
                    DEBUG_ID,
                    glow::DEBUG_SEVERITY_NOTIFICATION,
                    marker,
                );
            }
            C::PushDebugGroup(ref range) => {
                let marker = extract_marker(data, range);
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
        signal_fence: Option<(&mut super::Resource, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        for cmd_buf in command_buffers.iter() {
            for command in cmd_buf.commands.iter() {
                self.process(command, &cmd_buf.data);
            }
        }
        Ok(())
    }

    unsafe fn present(
        &mut self,
        surface: &mut super::Surface,
        texture: super::Texture,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }
}
