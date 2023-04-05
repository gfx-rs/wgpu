use super::Command as C;
use arrayvec::ArrayVec;
use glow::HasContext;
use std::{mem, slice, sync::Arc};

#[cfg(not(target_arch = "wasm32"))]
const DEBUG_ID: u32 = 0;

const CUBEMAP_FACES: [u32; 6] = [
    glow::TEXTURE_CUBE_MAP_POSITIVE_X,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_X,
    glow::TEXTURE_CUBE_MAP_POSITIVE_Y,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_Y,
    glow::TEXTURE_CUBE_MAP_POSITIVE_Z,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_Z,
];

#[cfg(not(target_arch = "wasm32"))]
fn extract_marker<'a>(data: &'a [u8], range: &std::ops::Range<u32>) -> &'a str {
    std::str::from_utf8(&data[range.start as usize..range.end as usize]).unwrap()
}

fn is_layered_target(target: super::BindTarget) -> bool {
    match target {
        glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D | glow::TEXTURE_CUBE_MAP_ARRAY => true,
        _ => false,
    }
}

impl super::Queue {
    /// Performs a manual shader clear, used as a workaround for a clearing bug on mesa
    unsafe fn perform_shader_clear(&self, gl: &glow::Context, draw_buffer: u32, color: [f32; 4]) {
        unsafe { gl.use_program(Some(self.shader_clear_program)) };
        unsafe {
            gl.uniform_4_f32(
                Some(&self.shader_clear_program_color_uniform_location),
                color[0],
                color[1],
                color[2],
                color[3],
            )
        };
        unsafe { gl.disable(glow::DEPTH_TEST) };
        unsafe { gl.disable(glow::STENCIL_TEST) };
        unsafe { gl.disable(glow::SCISSOR_TEST) };
        unsafe { gl.disable(glow::BLEND) };
        unsafe { gl.disable(glow::CULL_FACE) };
        unsafe { gl.draw_buffers(&[glow::COLOR_ATTACHMENT0 + draw_buffer]) };
        unsafe { gl.draw_arrays(glow::TRIANGLES, 0, 3) };

        if self.draw_buffer_count != 0 {
            // Reset the draw buffers to what they were before the clear
            let indices = (0..self.draw_buffer_count as u32)
                .map(|i| glow::COLOR_ATTACHMENT0 + i)
                .collect::<ArrayVec<_, { crate::MAX_COLOR_ATTACHMENTS }>>();
            unsafe { gl.draw_buffers(&indices) };
        }
        #[cfg(not(target_arch = "wasm32"))]
        for draw_buffer in 0..self.draw_buffer_count as u32 {
            unsafe { gl.disable_draw_buffer(glow::BLEND, draw_buffer) };
        }
    }

    unsafe fn reset_state(&mut self, gl: &glow::Context) {
        unsafe { gl.use_program(None) };
        unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, None) };
        unsafe { gl.disable(glow::DEPTH_TEST) };
        unsafe { gl.disable(glow::STENCIL_TEST) };
        unsafe { gl.disable(glow::SCISSOR_TEST) };
        unsafe { gl.disable(glow::BLEND) };
        unsafe { gl.disable(glow::CULL_FACE) };
        unsafe { gl.disable(glow::POLYGON_OFFSET_FILL) };
        unsafe { gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE) };
        if self.features.contains(wgt::Features::DEPTH_CLIP_CONTROL) {
            unsafe { gl.disable(glow::DEPTH_CLAMP) };
        }

        unsafe { gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None) };
        self.current_index_buffer = None;
    }

    unsafe fn set_attachment(
        &self,
        gl: &glow::Context,
        fbo_target: u32,
        attachment: u32,
        view: &super::TextureView,
    ) {
        match view.inner {
            super::TextureInner::Renderbuffer { raw } => {
                unsafe {
                    gl.framebuffer_renderbuffer(
                        fbo_target,
                        attachment,
                        glow::RENDERBUFFER,
                        Some(raw),
                    )
                };
            }
            super::TextureInner::DefaultRenderbuffer => panic!("Unexpected default RBO"),
            super::TextureInner::Texture { raw, target } => {
                let num_layers = view.array_layers.end - view.array_layers.start;
                if num_layers > 1 {
                    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
                    unsafe {
                        gl.framebuffer_texture_multiview_ovr(
                            fbo_target,
                            attachment,
                            Some(raw),
                            view.mip_levels.start as i32,
                            view.array_layers.start as i32,
                            num_layers as i32,
                        )
                    };
                } else if is_layered_target(target) {
                    unsafe {
                        gl.framebuffer_texture_layer(
                            fbo_target,
                            attachment,
                            Some(raw),
                            view.mip_levels.start as i32,
                            view.array_layers.start as i32,
                        )
                    };
                } else if target == glow::TEXTURE_CUBE_MAP {
                    unsafe {
                        gl.framebuffer_texture_2d(
                            fbo_target,
                            attachment,
                            CUBEMAP_FACES[view.array_layers.start as usize],
                            Some(raw),
                            view.mip_levels.start as i32,
                        )
                    };
                } else {
                    unsafe {
                        gl.framebuffer_texture_2d(
                            fbo_target,
                            attachment,
                            target,
                            Some(raw),
                            view.mip_levels.start as i32,
                        )
                    };
                }
            }
            #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
            super::TextureInner::ExternalFramebuffer { ref inner } => unsafe {
                gl.bind_external_framebuffer(glow::FRAMEBUFFER, inner);
            },
        }
    }

    unsafe fn process(
        &mut self,
        gl: &glow::Context,
        command: &C,
        #[cfg_attr(target_arch = "wasm32", allow(unused))] data_bytes: &[u8],
        queries: &[glow::Query],
    ) {
        match *command {
            C::Draw {
                topology,
                start_vertex,
                vertex_count,
                instance_count,
            } => {
                // Don't use `gl.draw_arrays` for `instance_count == 1`.
                // Angle has a bug where it doesn't consider the instance divisor when `DYNAMIC_DRAW` is used in `draw_arrays`.
                // See https://github.com/gfx-rs/wgpu/issues/3578
                unsafe {
                    gl.draw_arrays_instanced(
                        topology,
                        start_vertex as i32,
                        vertex_count as i32,
                        instance_count as i32,
                    )
                };
            }
            C::DrawIndexed {
                topology,
                index_type,
                index_count,
                index_offset,
                base_vertex,
                instance_count,
            } => {
                match base_vertex {
                    // Don't use `gl.draw_elements`/`gl.draw_elements_base_vertex` for `instance_count == 1`.
                    // Angle has a bug where it doesn't consider the instance divisor when `DYNAMIC_DRAW` is used in `gl.draw_elements`/`gl.draw_elements_base_vertex`.
                    // See https://github.com/gfx-rs/wgpu/issues/3578
                    0 => unsafe {
                        gl.draw_elements_instanced(
                            topology,
                            index_count as i32,
                            index_type,
                            index_offset as i32,
                            instance_count as i32,
                        )
                    },
                    _ => unsafe {
                        gl.draw_elements_instanced_base_vertex(
                            topology,
                            index_count as _,
                            index_type,
                            index_offset as i32,
                            instance_count as i32,
                            base_vertex,
                        )
                    },
                }
            }
            C::DrawIndirect {
                topology,
                indirect_buf,
                indirect_offset,
            } => {
                unsafe { gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf)) };
                unsafe { gl.draw_arrays_indirect_offset(topology, indirect_offset as i32) };
            }
            C::DrawIndexedIndirect {
                topology,
                index_type,
                indirect_buf,
                indirect_offset,
            } => {
                unsafe { gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf)) };
                unsafe {
                    gl.draw_elements_indirect_offset(topology, index_type, indirect_offset as i32)
                };
            }
            C::Dispatch(group_counts) => {
                unsafe { gl.dispatch_compute(group_counts[0], group_counts[1], group_counts[2]) };
            }
            C::DispatchIndirect {
                indirect_buf,
                indirect_offset,
            } => {
                unsafe { gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(indirect_buf)) };
                unsafe { gl.dispatch_compute_indirect(indirect_offset as i32) };
            }
            C::ClearBuffer {
                ref dst,
                dst_target,
                ref range,
            } => match dst.raw {
                Some(buffer) => {
                    // When `INDEX_BUFFER_ROLE_CHANGE` isn't available, we can't copy into the
                    // index buffer from the zero buffer. This would fail in Chrome with the
                    // following message:
                    //
                    // > Cannot copy into an element buffer destination from a non-element buffer
                    // > source
                    //
                    // Instead, we'll upload zeroes into the buffer.
                    let can_use_zero_buffer = self
                        .shared
                        .private_caps
                        .contains(super::PrivateCapabilities::INDEX_BUFFER_ROLE_CHANGE)
                        || dst_target != glow::ELEMENT_ARRAY_BUFFER;

                    if can_use_zero_buffer {
                        unsafe { gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.zero_buffer)) };
                        unsafe { gl.bind_buffer(dst_target, Some(buffer)) };
                        let mut dst_offset = range.start;
                        while dst_offset < range.end {
                            let size = (range.end - dst_offset).min(super::ZERO_BUFFER_SIZE as u64);
                            unsafe {
                                gl.copy_buffer_sub_data(
                                    glow::COPY_READ_BUFFER,
                                    dst_target,
                                    0,
                                    dst_offset as i32,
                                    size as i32,
                                )
                            };
                            dst_offset += size;
                        }
                    } else {
                        unsafe { gl.bind_buffer(dst_target, Some(buffer)) };
                        let zeroes = vec![0u8; (range.end - range.start) as usize];
                        unsafe {
                            gl.buffer_sub_data_u8_slice(dst_target, range.start as i32, &zeroes)
                        };
                    }
                }
                None => {
                    dst.data.as_ref().unwrap().lock().unwrap().as_mut_slice()
                        [range.start as usize..range.end as usize]
                        .fill(0);
                }
            },
            C::CopyBufferToBuffer {
                ref src,
                src_target,
                ref dst,
                dst_target,
                copy,
            } => {
                let copy_src_target = glow::COPY_READ_BUFFER;
                let is_index_buffer_only_element_dst = !self
                    .shared
                    .private_caps
                    .contains(super::PrivateCapabilities::INDEX_BUFFER_ROLE_CHANGE)
                    && dst_target == glow::ELEMENT_ARRAY_BUFFER
                    || src_target == glow::ELEMENT_ARRAY_BUFFER;

                // WebGL not allowed to copy data from other targets to element buffer and can't copy element data to other buffers
                let copy_dst_target = if is_index_buffer_only_element_dst {
                    glow::ELEMENT_ARRAY_BUFFER
                } else {
                    glow::COPY_WRITE_BUFFER
                };
                let size = copy.size.get() as usize;
                match (src.raw, dst.raw) {
                    (Some(ref src), Some(ref dst)) => {
                        unsafe { gl.bind_buffer(copy_src_target, Some(*src)) };
                        unsafe { gl.bind_buffer(copy_dst_target, Some(*dst)) };
                        unsafe {
                            gl.copy_buffer_sub_data(
                                copy_src_target,
                                copy_dst_target,
                                copy.src_offset as _,
                                copy.dst_offset as _,
                                copy.size.get() as _,
                            )
                        };
                    }
                    (Some(src), None) => {
                        let mut data = dst.data.as_ref().unwrap().lock().unwrap();
                        let dst_data = &mut data.as_mut_slice()
                            [copy.dst_offset as usize..copy.dst_offset as usize + size];

                        unsafe { gl.bind_buffer(copy_src_target, Some(src)) };
                        unsafe {
                            self.shared.get_buffer_sub_data(
                                gl,
                                copy_src_target,
                                copy.src_offset as i32,
                                dst_data,
                            )
                        };
                    }
                    (None, Some(dst)) => {
                        let data = src.data.as_ref().unwrap().lock().unwrap();
                        let src_data = &data.as_slice()
                            [copy.src_offset as usize..copy.src_offset as usize + size];
                        unsafe { gl.bind_buffer(copy_dst_target, Some(dst)) };
                        unsafe {
                            gl.buffer_sub_data_u8_slice(
                                copy_dst_target,
                                copy.dst_offset as i32,
                                src_data,
                            )
                        };
                    }
                    (None, None) => {
                        todo!()
                    }
                }
                unsafe { gl.bind_buffer(copy_src_target, None) };
                if is_index_buffer_only_element_dst {
                    unsafe {
                        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, self.current_index_buffer)
                    };
                } else {
                    unsafe { gl.bind_buffer(copy_dst_target, None) };
                }
            }
            #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
            C::CopyExternalImageToTexture {
                ref src,
                dst,
                dst_target,
                dst_format,
                dst_premultiplication,
                ref copy,
            } => {
                const UNPACK_FLIP_Y_WEBGL: u32 =
                    web_sys::WebGl2RenderingContext::UNPACK_FLIP_Y_WEBGL;
                const UNPACK_PREMULTIPLY_ALPHA_WEBGL: u32 =
                    web_sys::WebGl2RenderingContext::UNPACK_PREMULTIPLY_ALPHA_WEBGL;

                unsafe {
                    if src.flip_y {
                        gl.pixel_store_bool(UNPACK_FLIP_Y_WEBGL, true);
                    }
                    if dst_premultiplication {
                        gl.pixel_store_bool(UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
                    }
                }

                unsafe { gl.bind_texture(dst_target, Some(dst)) };
                let format_desc = self.shared.describe_texture_format(dst_format);
                if is_layered_target(dst_target) {
                    let z_offset =
                        if let glow::TEXTURE_2D_ARRAY | glow::TEXTURE_CUBE_MAP_ARRAY = dst_target {
                            copy.dst_base.array_layer as i32
                        } else {
                            copy.dst_base.origin.z as i32
                        };

                    match src.source {
                        wgt::ExternalImageSource::ImageBitmap(ref b) => unsafe {
                            gl.tex_sub_image_3d_with_image_bitmap(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                z_offset,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                copy.size.depth as i32,
                                format_desc.external,
                                format_desc.data_type,
                                b,
                            );
                        },
                        wgt::ExternalImageSource::HTMLVideoElement(ref v) => unsafe {
                            gl.tex_sub_image_3d_with_html_video_element(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                z_offset,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                copy.size.depth as i32,
                                format_desc.external,
                                format_desc.data_type,
                                v,
                            );
                        },
                        wgt::ExternalImageSource::HTMLCanvasElement(ref c) => unsafe {
                            gl.tex_sub_image_3d_with_html_canvas_element(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                z_offset,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                copy.size.depth as i32,
                                format_desc.external,
                                format_desc.data_type,
                                c,
                            );
                        },
                        wgt::ExternalImageSource::OffscreenCanvas(_) => unreachable!(),
                    }
                } else {
                    let dst_target = if let glow::TEXTURE_CUBE_MAP = dst_target {
                        CUBEMAP_FACES[copy.dst_base.array_layer as usize]
                    } else {
                        dst_target
                    };

                    match src.source {
                        wgt::ExternalImageSource::ImageBitmap(ref b) => unsafe {
                            gl.tex_sub_image_2d_with_image_bitmap_and_width_and_height(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                format_desc.external,
                                format_desc.data_type,
                                b,
                            );
                        },
                        wgt::ExternalImageSource::HTMLVideoElement(ref v) => unsafe {
                            gl.tex_sub_image_2d_with_html_video_and_width_and_height(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                format_desc.external,
                                format_desc.data_type,
                                v,
                            )
                        },
                        wgt::ExternalImageSource::HTMLCanvasElement(ref c) => unsafe {
                            gl.tex_sub_image_2d_with_html_canvas_and_width_and_height(
                                dst_target,
                                copy.dst_base.mip_level as i32,
                                copy.dst_base.origin.x as i32,
                                copy.dst_base.origin.y as i32,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                format_desc.external,
                                format_desc.data_type,
                                c,
                            )
                        },
                        wgt::ExternalImageSource::OffscreenCanvas(_) => unreachable!(),
                    }
                }

                unsafe {
                    if src.flip_y {
                        gl.pixel_store_bool(UNPACK_FLIP_Y_WEBGL, false);
                    }
                    if dst_premultiplication {
                        gl.pixel_store_bool(UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                    }
                }
            }
            C::CopyTextureToTexture {
                src,
                src_target,
                dst,
                dst_target,
                dst_is_cubemap,
                ref copy,
            } => {
                //TODO: handle 3D copies
                unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.copy_fbo)) };
                if is_layered_target(src_target) {
                    //TODO: handle GLES without framebuffer_texture_3d
                    unsafe {
                        gl.framebuffer_texture_layer(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            Some(src),
                            copy.src_base.mip_level as i32,
                            copy.src_base.array_layer as i32,
                        )
                    };
                } else {
                    unsafe {
                        gl.framebuffer_texture_2d(
                            glow::READ_FRAMEBUFFER,
                            glow::COLOR_ATTACHMENT0,
                            src_target,
                            Some(src),
                            copy.src_base.mip_level as i32,
                        )
                    };
                }

                unsafe { gl.bind_texture(dst_target, Some(dst)) };
                if dst_is_cubemap {
                    unsafe {
                        gl.copy_tex_sub_image_2d(
                            CUBEMAP_FACES[copy.dst_base.array_layer as usize],
                            copy.dst_base.mip_level as i32,
                            copy.dst_base.origin.x as i32,
                            copy.dst_base.origin.y as i32,
                            copy.src_base.origin.x as i32,
                            copy.src_base.origin.y as i32,
                            copy.size.width as i32,
                            copy.size.height as i32,
                        )
                    };
                } else if is_layered_target(dst_target) {
                    unsafe {
                        gl.copy_tex_sub_image_3d(
                            dst_target,
                            copy.dst_base.mip_level as i32,
                            copy.dst_base.origin.x as i32,
                            copy.dst_base.origin.y as i32,
                            if let glow::TEXTURE_2D_ARRAY | glow::TEXTURE_CUBE_MAP_ARRAY =
                                dst_target
                            {
                                copy.dst_base.array_layer as i32
                            } else {
                                copy.dst_base.origin.z as i32
                            },
                            copy.src_base.origin.x as i32,
                            copy.src_base.origin.y as i32,
                            copy.size.width as i32,
                            copy.size.height as i32,
                        )
                    };
                } else {
                    unsafe {
                        gl.copy_tex_sub_image_2d(
                            dst_target,
                            copy.dst_base.mip_level as i32,
                            copy.dst_base.origin.x as i32,
                            copy.dst_base.origin.y as i32,
                            copy.src_base.origin.x as i32,
                            copy.src_base.origin.y as i32,
                            copy.size.width as i32,
                            copy.size.height as i32,
                        )
                    };
                }
            }
            C::CopyBufferToTexture {
                ref src,
                src_target: _,
                dst,
                dst_target,
                dst_format,
                ref copy,
            } => {
                let (block_width, block_height) = dst_format.block_dimensions();
                let block_size = dst_format.block_size(None).unwrap();
                let format_desc = self.shared.describe_texture_format(dst_format);
                let row_texels = copy
                    .buffer_layout
                    .bytes_per_row
                    .map_or(0, |bpr| block_width * bpr / block_size);
                let column_texels = copy
                    .buffer_layout
                    .rows_per_image
                    .map_or(0, |rpi| block_height * rpi);

                unsafe { gl.bind_texture(dst_target, Some(dst)) };
                unsafe { gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, row_texels as i32) };
                unsafe { gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, column_texels as i32) };
                let mut unbind_unpack_buffer = false;
                if !dst_format.is_compressed() {
                    let buffer_data;
                    let unpack_data = match src.raw {
                        Some(buffer) => {
                            unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer)) };
                            unbind_unpack_buffer = true;
                            glow::PixelUnpackData::BufferOffset(copy.buffer_layout.offset as u32)
                        }
                        None => {
                            buffer_data = src.data.as_ref().unwrap().lock().unwrap();
                            let src_data =
                                &buffer_data.as_slice()[copy.buffer_layout.offset as usize..];
                            glow::PixelUnpackData::Slice(src_data)
                        }
                    };
                    if is_layered_target(dst_target) {
                        unsafe {
                            gl.tex_sub_image_3d(
                                dst_target,
                                copy.texture_base.mip_level as i32,
                                copy.texture_base.origin.x as i32,
                                copy.texture_base.origin.y as i32,
                                if let glow::TEXTURE_2D_ARRAY | glow::TEXTURE_CUBE_MAP_ARRAY =
                                    dst_target
                                {
                                    copy.texture_base.array_layer as i32
                                } else {
                                    copy.texture_base.origin.z as i32
                                },
                                copy.size.width as i32,
                                copy.size.height as i32,
                                copy.size.depth as i32,
                                format_desc.external,
                                format_desc.data_type,
                                unpack_data,
                            )
                        };
                    } else {
                        unsafe {
                            gl.tex_sub_image_2d(
                                if let glow::TEXTURE_CUBE_MAP = dst_target {
                                    CUBEMAP_FACES[copy.texture_base.array_layer as usize]
                                } else {
                                    dst_target
                                },
                                copy.texture_base.mip_level as i32,
                                copy.texture_base.origin.x as i32,
                                copy.texture_base.origin.y as i32,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                format_desc.external,
                                format_desc.data_type,
                                unpack_data,
                            )
                        };
                    }
                } else {
                    let bytes_per_row = copy
                        .buffer_layout
                        .bytes_per_row
                        .unwrap_or(copy.size.width * block_size);
                    let minimum_rows_per_image =
                        (copy.size.height + block_height - 1) / block_height;
                    let rows_per_image = copy
                        .buffer_layout
                        .rows_per_image
                        .unwrap_or(minimum_rows_per_image);

                    let bytes_per_image = bytes_per_row * rows_per_image;
                    let minimum_bytes_per_image = bytes_per_row * minimum_rows_per_image;
                    let bytes_in_upload =
                        (bytes_per_image * (copy.size.depth - 1)) + minimum_bytes_per_image;
                    let offset = copy.buffer_layout.offset as u32;

                    let buffer_data;
                    let unpack_data = match src.raw {
                        Some(buffer) => {
                            unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer)) };
                            unbind_unpack_buffer = true;
                            glow::CompressedPixelUnpackData::BufferRange(
                                offset..offset + bytes_in_upload,
                            )
                        }
                        None => {
                            buffer_data = src.data.as_ref().unwrap().lock().unwrap();
                            let src_data = &buffer_data.as_slice()
                                [(offset as usize)..(offset + bytes_in_upload) as usize];
                            glow::CompressedPixelUnpackData::Slice(src_data)
                        }
                    };

                    if is_layered_target(dst_target) {
                        unsafe {
                            gl.compressed_tex_sub_image_3d(
                                dst_target,
                                copy.texture_base.mip_level as i32,
                                copy.texture_base.origin.x as i32,
                                copy.texture_base.origin.y as i32,
                                if let glow::TEXTURE_2D_ARRAY | glow::TEXTURE_CUBE_MAP_ARRAY =
                                    dst_target
                                {
                                    copy.texture_base.array_layer as i32
                                } else {
                                    copy.texture_base.origin.z as i32
                                },
                                copy.size.width as i32,
                                copy.size.height as i32,
                                copy.size.depth as i32,
                                format_desc.internal,
                                unpack_data,
                            )
                        };
                    } else {
                        unsafe {
                            gl.compressed_tex_sub_image_2d(
                                if let glow::TEXTURE_CUBE_MAP = dst_target {
                                    CUBEMAP_FACES[copy.texture_base.array_layer as usize]
                                } else {
                                    dst_target
                                },
                                copy.texture_base.mip_level as i32,
                                copy.texture_base.origin.x as i32,
                                copy.texture_base.origin.y as i32,
                                copy.size.width as i32,
                                copy.size.height as i32,
                                format_desc.internal,
                                unpack_data,
                            )
                        };
                    }
                }
                if unbind_unpack_buffer {
                    unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None) };
                }
            }
            C::CopyTextureToBuffer {
                src,
                src_target,
                src_format,
                ref dst,
                dst_target: _,
                ref copy,
            } => {
                let block_size = src_format.block_size(None).unwrap();
                if src_format.is_compressed() {
                    log::error!("Not implemented yet: compressed texture copy to buffer");
                    return;
                }
                if src_target == glow::TEXTURE_CUBE_MAP
                    || src_target == glow::TEXTURE_CUBE_MAP_ARRAY
                {
                    log::error!("Not implemented yet: cubemap texture copy to buffer");
                    return;
                }
                let format_desc = self.shared.describe_texture_format(src_format);
                let row_texels = copy
                    .buffer_layout
                    .bytes_per_row
                    .map_or(copy.size.width, |bpr| bpr / block_size);
                let column_texels = copy
                    .buffer_layout
                    .rows_per_image
                    .unwrap_or(copy.size.height);

                unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.copy_fbo)) };

                let read_pixels = |offset| {
                    let mut buffer_data;
                    let unpack_data = match dst.raw {
                        Some(buffer) => {
                            unsafe { gl.pixel_store_i32(glow::PACK_ROW_LENGTH, row_texels as i32) };
                            unsafe { gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer)) };
                            glow::PixelPackData::BufferOffset(offset as u32)
                        }
                        None => {
                            buffer_data = dst.data.as_ref().unwrap().lock().unwrap();
                            let dst_data = &mut buffer_data.as_mut_slice()[offset as usize..];
                            glow::PixelPackData::Slice(dst_data)
                        }
                    };
                    unsafe {
                        gl.read_pixels(
                            copy.texture_base.origin.x as i32,
                            copy.texture_base.origin.y as i32,
                            copy.size.width as i32,
                            copy.size.height as i32,
                            format_desc.external,
                            format_desc.data_type,
                            unpack_data,
                        )
                    };
                };

                match src_target {
                    glow::TEXTURE_2D => {
                        unsafe {
                            gl.framebuffer_texture_2d(
                                glow::READ_FRAMEBUFFER,
                                glow::COLOR_ATTACHMENT0,
                                src_target,
                                Some(src),
                                copy.texture_base.mip_level as i32,
                            )
                        };
                        read_pixels(copy.buffer_layout.offset);
                    }
                    glow::TEXTURE_2D_ARRAY => {
                        unsafe {
                            gl.framebuffer_texture_layer(
                                glow::READ_FRAMEBUFFER,
                                glow::COLOR_ATTACHMENT0,
                                Some(src),
                                copy.texture_base.mip_level as i32,
                                copy.texture_base.array_layer as i32,
                            )
                        };
                        read_pixels(copy.buffer_layout.offset);
                    }
                    glow::TEXTURE_3D => {
                        for z in copy.texture_base.origin.z..copy.size.depth {
                            unsafe {
                                gl.framebuffer_texture_layer(
                                    glow::READ_FRAMEBUFFER,
                                    glow::COLOR_ATTACHMENT0,
                                    Some(src),
                                    copy.texture_base.mip_level as i32,
                                    z as i32,
                                )
                            };
                            let offset = copy.buffer_layout.offset
                                + (z * block_size * row_texels * column_texels) as u64;
                            read_pixels(offset);
                        }
                    }
                    glow::TEXTURE_CUBE_MAP | glow::TEXTURE_CUBE_MAP_ARRAY => unimplemented!(),
                    _ => unreachable!(),
                }
            }
            C::SetIndexBuffer(buffer) => {
                unsafe { gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(buffer)) };
                self.current_index_buffer = Some(buffer);
            }
            C::BeginQuery(query, target) => {
                unsafe { gl.begin_query(target, query) };
            }
            C::EndQuery(target) => {
                unsafe { gl.end_query(target) };
            }
            C::CopyQueryResults {
                ref query_range,
                ref dst,
                dst_target,
                dst_offset,
            } => {
                self.temp_query_results.clear();
                for &query in queries[query_range.start as usize..query_range.end as usize].iter() {
                    let result = unsafe { gl.get_query_parameter_u32(query, glow::QUERY_RESULT) };
                    self.temp_query_results.push(result as u64);
                }
                let query_data = unsafe {
                    slice::from_raw_parts(
                        self.temp_query_results.as_ptr() as *const u8,
                        self.temp_query_results.len() * mem::size_of::<u64>(),
                    )
                };
                match dst.raw {
                    Some(buffer) => {
                        unsafe { gl.bind_buffer(dst_target, Some(buffer)) };
                        unsafe {
                            gl.buffer_sub_data_u8_slice(dst_target, dst_offset as i32, query_data)
                        };
                    }
                    None => {
                        let data = &mut dst.data.as_ref().unwrap().lock().unwrap();
                        let len = query_data.len().min(data.len());
                        data[..len].copy_from_slice(&query_data[..len]);
                    }
                }
            }
            C::ResetFramebuffer { is_default } => {
                if is_default {
                    unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
                } else {
                    unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.draw_fbo)) };
                    unsafe {
                        gl.framebuffer_texture_2d(
                            glow::DRAW_FRAMEBUFFER,
                            glow::DEPTH_STENCIL_ATTACHMENT,
                            glow::TEXTURE_2D,
                            None,
                            0,
                        )
                    };
                    for i in 0..crate::MAX_COLOR_ATTACHMENTS {
                        let target = glow::COLOR_ATTACHMENT0 + i as u32;
                        unsafe {
                            gl.framebuffer_texture_2d(
                                glow::DRAW_FRAMEBUFFER,
                                target,
                                glow::TEXTURE_2D,
                                None,
                                0,
                            )
                        };
                    }
                }
                unsafe { gl.color_mask(true, true, true, true) };
                unsafe { gl.depth_mask(true) };
                unsafe { gl.stencil_mask(!0) };
                unsafe { gl.disable(glow::DEPTH_TEST) };
                unsafe { gl.disable(glow::STENCIL_TEST) };
                unsafe { gl.disable(glow::SCISSOR_TEST) };
            }
            C::BindAttachment {
                attachment,
                ref view,
            } => {
                unsafe { self.set_attachment(gl, glow::DRAW_FRAMEBUFFER, attachment, view) };
            }
            C::ResolveAttachment {
                attachment,
                ref dst,
                ref size,
            } => {
                unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.draw_fbo)) };
                unsafe { gl.read_buffer(attachment) };
                unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.copy_fbo)) };
                unsafe {
                    self.set_attachment(gl, glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, dst)
                };
                unsafe {
                    gl.blit_framebuffer(
                        0,
                        0,
                        size.width as i32,
                        size.height as i32,
                        0,
                        0,
                        size.width as i32,
                        size.height as i32,
                        glow::COLOR_BUFFER_BIT,
                        glow::NEAREST,
                    )
                };
                unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };
                unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.draw_fbo)) };
            }
            C::InvalidateAttachments(ref list) => {
                unsafe { gl.invalidate_framebuffer(glow::DRAW_FRAMEBUFFER, list) };
            }
            C::SetDrawColorBuffers(count) => {
                self.draw_buffer_count = count;
                let indices = (0..count as u32)
                    .map(|i| glow::COLOR_ATTACHMENT0 + i)
                    .collect::<ArrayVec<_, { crate::MAX_COLOR_ATTACHMENTS }>>();
                unsafe { gl.draw_buffers(&indices) };

                if self
                    .shared
                    .private_caps
                    .contains(super::PrivateCapabilities::CAN_DISABLE_DRAW_BUFFER)
                {
                    for draw_buffer in 0..count as u32 {
                        unsafe { gl.disable_draw_buffer(glow::BLEND, draw_buffer) };
                    }
                }
            }
            C::ClearColorF {
                draw_buffer,
                ref color,
                is_srgb,
            } => {
                if self
                    .shared
                    .workarounds
                    .contains(super::Workarounds::MESA_I915_SRGB_SHADER_CLEAR)
                    && is_srgb
                {
                    unsafe { self.perform_shader_clear(gl, draw_buffer, *color) };
                } else {
                    unsafe { gl.clear_buffer_f32_slice(glow::COLOR, draw_buffer, color) };
                }
            }
            C::ClearColorU(draw_buffer, ref color) => {
                unsafe { gl.clear_buffer_u32_slice(glow::COLOR, draw_buffer, color) };
            }
            C::ClearColorI(draw_buffer, ref color) => {
                unsafe { gl.clear_buffer_i32_slice(glow::COLOR, draw_buffer, color) };
            }
            C::ClearDepth(depth) => {
                unsafe { gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[depth]) };
            }
            C::ClearStencil(value) => {
                unsafe { gl.clear_buffer_i32_slice(glow::STENCIL, 0, &[value as i32]) };
            }
            C::ClearDepthAndStencil(depth, stencil_value) => {
                unsafe {
                    gl.clear_buffer_depth_stencil(
                        glow::DEPTH_STENCIL,
                        0,
                        depth,
                        stencil_value as i32,
                    )
                };
            }
            C::BufferBarrier(raw, usage) => {
                let mut flags = 0;
                if usage.contains(crate::BufferUses::VERTEX) {
                    flags |= glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
                    unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, Some(raw)) };
                    unsafe { gl.vertex_attrib_pointer_f32(0, 1, glow::BYTE, true, 0, 0) };
                }
                if usage.contains(crate::BufferUses::INDEX) {
                    flags |= glow::ELEMENT_ARRAY_BARRIER_BIT;
                    unsafe { gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(raw)) };
                }
                if usage.contains(crate::BufferUses::UNIFORM) {
                    flags |= glow::UNIFORM_BARRIER_BIT;
                }
                if usage.contains(crate::BufferUses::INDIRECT) {
                    flags |= glow::COMMAND_BARRIER_BIT;
                    unsafe { gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(raw)) };
                }
                if usage.contains(crate::BufferUses::COPY_SRC) {
                    flags |= glow::PIXEL_BUFFER_BARRIER_BIT;
                    unsafe { gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(raw)) };
                }
                if usage.contains(crate::BufferUses::COPY_DST) {
                    flags |= glow::PIXEL_BUFFER_BARRIER_BIT;
                    unsafe { gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(raw)) };
                }
                if usage.intersects(crate::BufferUses::MAP_READ | crate::BufferUses::MAP_WRITE) {
                    flags |= glow::BUFFER_UPDATE_BARRIER_BIT;
                }
                if usage.intersects(
                    crate::BufferUses::STORAGE_READ | crate::BufferUses::STORAGE_READ_WRITE,
                ) {
                    flags |= glow::SHADER_STORAGE_BARRIER_BIT;
                }
                unsafe { gl.memory_barrier(flags) };
            }
            C::TextureBarrier(usage) => {
                let mut flags = 0;
                if usage.contains(crate::TextureUses::RESOURCE) {
                    flags |= glow::TEXTURE_FETCH_BARRIER_BIT;
                }
                if usage.intersects(
                    crate::TextureUses::STORAGE_READ | crate::TextureUses::STORAGE_READ_WRITE,
                ) {
                    flags |= glow::SHADER_IMAGE_ACCESS_BARRIER_BIT;
                }
                if usage.contains(crate::TextureUses::COPY_DST) {
                    flags |= glow::TEXTURE_UPDATE_BARRIER_BIT;
                }
                if usage.intersects(
                    crate::TextureUses::COLOR_TARGET
                        | crate::TextureUses::DEPTH_STENCIL_READ
                        | crate::TextureUses::DEPTH_STENCIL_WRITE,
                ) {
                    flags |= glow::FRAMEBUFFER_BARRIER_BIT;
                }
                unsafe { gl.memory_barrier(flags) };
            }
            C::SetViewport {
                ref rect,
                ref depth,
            } => {
                unsafe { gl.viewport(rect.x, rect.y, rect.w, rect.h) };
                unsafe { gl.depth_range_f32(depth.start, depth.end) };
            }
            C::SetScissor(ref rect) => {
                unsafe { gl.scissor(rect.x, rect.y, rect.w, rect.h) };
                unsafe { gl.enable(glow::SCISSOR_TEST) };
            }
            C::SetStencilFunc {
                face,
                function,
                reference,
                read_mask,
            } => {
                unsafe { gl.stencil_func_separate(face, function, reference as i32, read_mask) };
            }
            C::SetStencilOps {
                face,
                write_mask,
                ref ops,
            } => {
                unsafe { gl.stencil_mask_separate(face, write_mask) };
                unsafe { gl.stencil_op_separate(face, ops.fail, ops.depth_fail, ops.pass) };
            }
            C::SetVertexAttribute {
                buffer,
                ref buffer_desc,
                attribute_desc: ref vat,
            } => {
                unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, buffer) };
                unsafe { gl.enable_vertex_attrib_array(vat.location) };

                if buffer.is_none() {
                    match vat.format_desc.attrib_kind {
                        super::VertexAttribKind::Float => unsafe {
                            gl.vertex_attrib_format_f32(
                                vat.location,
                                vat.format_desc.element_count,
                                vat.format_desc.element_format,
                                true, // always normalized
                                vat.offset,
                            )
                        },
                        super::VertexAttribKind::Integer => unsafe {
                            gl.vertex_attrib_format_i32(
                                vat.location,
                                vat.format_desc.element_count,
                                vat.format_desc.element_format,
                                vat.offset,
                            )
                        },
                    }

                    //Note: there is apparently a bug on AMD 3500U:
                    // this call is ignored if the current array is disabled.
                    unsafe { gl.vertex_attrib_binding(vat.location, vat.buffer_index) };
                } else {
                    match vat.format_desc.attrib_kind {
                        super::VertexAttribKind::Float => unsafe {
                            gl.vertex_attrib_pointer_f32(
                                vat.location,
                                vat.format_desc.element_count,
                                vat.format_desc.element_format,
                                true, // always normalized
                                buffer_desc.stride as i32,
                                vat.offset as i32,
                            )
                        },
                        super::VertexAttribKind::Integer => unsafe {
                            gl.vertex_attrib_pointer_i32(
                                vat.location,
                                vat.format_desc.element_count,
                                vat.format_desc.element_format,
                                buffer_desc.stride as i32,
                                vat.offset as i32,
                            )
                        },
                    }
                    unsafe { gl.vertex_attrib_divisor(vat.location, buffer_desc.step as u32) };
                }
            }
            C::UnsetVertexAttribute(location) => {
                unsafe { gl.disable_vertex_attrib_array(location) };
            }
            C::SetVertexBuffer {
                index,
                ref buffer,
                ref buffer_desc,
            } => {
                unsafe { gl.vertex_binding_divisor(index, buffer_desc.step as u32) };
                unsafe {
                    gl.bind_vertex_buffer(
                        index,
                        Some(buffer.raw),
                        buffer.offset as i32,
                        buffer_desc.stride as i32,
                    )
                };
            }
            C::SetDepth(ref depth) => {
                unsafe { gl.depth_func(depth.function) };
                unsafe { gl.depth_mask(depth.mask) };
            }
            C::SetDepthBias(bias) => {
                if bias.is_enabled() {
                    unsafe { gl.enable(glow::POLYGON_OFFSET_FILL) };
                    unsafe { gl.polygon_offset(bias.constant as f32, bias.slope_scale) };
                } else {
                    unsafe { gl.disable(glow::POLYGON_OFFSET_FILL) };
                }
            }
            C::ConfigureDepthStencil(aspects) => {
                if aspects.contains(crate::FormatAspects::DEPTH) {
                    unsafe { gl.enable(glow::DEPTH_TEST) };
                } else {
                    unsafe { gl.disable(glow::DEPTH_TEST) };
                }
                if aspects.contains(crate::FormatAspects::STENCIL) {
                    unsafe { gl.enable(glow::STENCIL_TEST) };
                } else {
                    unsafe { gl.disable(glow::STENCIL_TEST) };
                }
            }
            C::SetAlphaToCoverage(enabled) => {
                if enabled {
                    unsafe { gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE) };
                } else {
                    unsafe { gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE) };
                }
            }
            C::SetProgram(program) => {
                unsafe { gl.use_program(Some(program)) };
            }
            C::SetPrimitive(ref state) => {
                unsafe { gl.front_face(state.front_face) };
                if state.cull_face != 0 {
                    unsafe { gl.enable(glow::CULL_FACE) };
                    unsafe { gl.cull_face(state.cull_face) };
                } else {
                    unsafe { gl.disable(glow::CULL_FACE) };
                }
                if self.features.contains(wgt::Features::DEPTH_CLIP_CONTROL) {
                    //Note: this is a bit tricky, since we are controlling the clip, not the clamp.
                    if state.unclipped_depth {
                        unsafe { gl.enable(glow::DEPTH_CLAMP) };
                    } else {
                        unsafe { gl.disable(glow::DEPTH_CLAMP) };
                    }
                }
            }
            C::SetBlendConstant(c) => {
                unsafe { gl.blend_color(c[0], c[1], c[2], c[3]) };
            }
            C::SetColorTarget {
                draw_buffer_index,
                desc: super::ColorTargetDesc { mask, ref blend },
            } => {
                use wgt::ColorWrites as Cw;
                if let Some(index) = draw_buffer_index {
                    unsafe {
                        gl.color_mask_draw_buffer(
                            index,
                            mask.contains(Cw::RED),
                            mask.contains(Cw::GREEN),
                            mask.contains(Cw::BLUE),
                            mask.contains(Cw::ALPHA),
                        )
                    };
                    if let Some(ref blend) = *blend {
                        unsafe { gl.enable_draw_buffer(index, glow::BLEND) };
                        if blend.color != blend.alpha {
                            unsafe {
                                gl.blend_equation_separate_draw_buffer(
                                    index,
                                    blend.color.equation,
                                    blend.alpha.equation,
                                )
                            };
                            unsafe {
                                gl.blend_func_separate_draw_buffer(
                                    index,
                                    blend.color.src,
                                    blend.color.dst,
                                    blend.alpha.src,
                                    blend.alpha.dst,
                                )
                            };
                        } else {
                            unsafe { gl.blend_equation_draw_buffer(index, blend.color.equation) };
                            unsafe {
                                gl.blend_func_draw_buffer(index, blend.color.src, blend.color.dst)
                            };
                        }
                    } else if self
                        .shared
                        .private_caps
                        .contains(super::PrivateCapabilities::CAN_DISABLE_DRAW_BUFFER)
                    {
                        unsafe { gl.disable_draw_buffer(index, glow::BLEND) };
                    }
                } else {
                    unsafe {
                        gl.color_mask(
                            mask.contains(Cw::RED),
                            mask.contains(Cw::GREEN),
                            mask.contains(Cw::BLUE),
                            mask.contains(Cw::ALPHA),
                        )
                    };
                    if let Some(ref blend) = *blend {
                        unsafe { gl.enable(glow::BLEND) };
                        if blend.color != blend.alpha {
                            unsafe {
                                gl.blend_equation_separate(
                                    blend.color.equation,
                                    blend.alpha.equation,
                                )
                            };
                            unsafe {
                                gl.blend_func_separate(
                                    blend.color.src,
                                    blend.color.dst,
                                    blend.alpha.src,
                                    blend.alpha.dst,
                                )
                            };
                        } else {
                            unsafe { gl.blend_equation(blend.color.equation) };
                            unsafe { gl.blend_func(blend.color.src, blend.color.dst) };
                        }
                    } else {
                        unsafe { gl.disable(glow::BLEND) };
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
                unsafe { gl.bind_buffer_range(target, slot, Some(buffer), offset, size) };
            }
            C::BindSampler(texture_index, sampler) => {
                unsafe { gl.bind_sampler(texture_index, sampler) };
            }
            C::BindTexture {
                slot,
                texture,
                target,
                aspects,
            } => {
                unsafe { gl.active_texture(glow::TEXTURE0 + slot) };
                unsafe { gl.bind_texture(target, Some(texture)) };

                let version = gl.version();
                let is_min_es_3_1 = version.is_embedded && (version.major, version.minor) >= (3, 1);
                let is_min_4_3 = !version.is_embedded && (version.major, version.minor) >= (4, 3);
                if is_min_es_3_1 || is_min_4_3 {
                    let mode = match aspects {
                        crate::FormatAspects::DEPTH => Some(glow::DEPTH_COMPONENT),
                        crate::FormatAspects::STENCIL => Some(glow::STENCIL_INDEX),
                        _ => None,
                    };
                    if let Some(mode) = mode {
                        unsafe {
                            gl.tex_parameter_i32(
                                target,
                                glow::DEPTH_STENCIL_TEXTURE_MODE,
                                mode as _,
                            )
                        };
                    }
                }
            }
            C::BindImage { slot, ref binding } => {
                unsafe {
                    gl.bind_image_texture(
                        slot,
                        binding.raw,
                        binding.mip_level as i32,
                        binding.array_layer.is_none(),
                        binding.array_layer.unwrap_or_default() as i32,
                        binding.access,
                        binding.format,
                    )
                };
            }
            #[cfg(not(target_arch = "wasm32"))]
            C::InsertDebugMarker(ref range) => {
                let marker = extract_marker(data_bytes, range);
                unsafe {
                    gl.debug_message_insert(
                        glow::DEBUG_SOURCE_APPLICATION,
                        glow::DEBUG_TYPE_MARKER,
                        DEBUG_ID,
                        glow::DEBUG_SEVERITY_NOTIFICATION,
                        marker,
                    )
                };
            }
            #[cfg(target_arch = "wasm32")]
            C::InsertDebugMarker(_) => (),
            #[cfg_attr(target_arch = "wasm32", allow(unused))]
            C::PushDebugGroup(ref range) => {
                #[cfg(not(target_arch = "wasm32"))]
                let marker = extract_marker(data_bytes, range);
                #[cfg(not(target_arch = "wasm32"))]
                unsafe {
                    gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, DEBUG_ID, marker)
                };
            }
            C::PopDebugGroup => {
                #[cfg(not(target_arch = "wasm32"))]
                unsafe {
                    gl.pop_debug_group()
                };
            }
            C::SetPushConstants {
                ref uniform,
                offset,
            } => {
                fn get_data<T>(data: &[u8], offset: u32) -> &[T] {
                    let raw = &data[(offset as usize)..];
                    unsafe {
                        slice::from_raw_parts(
                            raw.as_ptr() as *const _,
                            raw.len() / mem::size_of::<T>(),
                        )
                    }
                }

                let location = uniform.location.as_ref();

                match uniform.utype {
                    glow::FLOAT => {
                        let data = get_data::<f32>(data_bytes, offset)[0];
                        unsafe { gl.uniform_1_f32(location, data) };
                    }
                    glow::FLOAT_VEC2 => {
                        let data = get_data::<[f32; 2]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_2_f32_slice(location, &data) };
                    }
                    glow::FLOAT_VEC3 => {
                        let data = get_data::<[f32; 3]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_3_f32_slice(location, &data) };
                    }
                    glow::FLOAT_VEC4 => {
                        let data = get_data::<[f32; 4]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_4_f32_slice(location, &data) };
                    }
                    glow::INT => {
                        let data = get_data::<i32>(data_bytes, offset)[0];
                        unsafe { gl.uniform_1_i32(location, data) };
                    }
                    glow::INT_VEC2 => {
                        let data = get_data::<[i32; 2]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_2_i32_slice(location, &data) };
                    }
                    glow::INT_VEC3 => {
                        let data = get_data::<[i32; 3]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_3_i32_slice(location, &data) };
                    }
                    glow::INT_VEC4 => {
                        let data = get_data::<[i32; 4]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_4_i32_slice(location, &data) };
                    }
                    glow::FLOAT_MAT2 => {
                        let data = get_data::<[f32; 4]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_matrix_2_f32_slice(location, false, &data) };
                    }
                    glow::FLOAT_MAT3 => {
                        let data = get_data::<[f32; 9]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_matrix_3_f32_slice(location, false, &data) };
                    }
                    glow::FLOAT_MAT4 => {
                        let data = get_data::<[f32; 16]>(data_bytes, offset)[0];
                        unsafe { gl.uniform_matrix_4_f32_slice(location, false, &data) };
                    }
                    _ => panic!("Unsupported uniform datatype!"),
                }
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
        let shared = Arc::clone(&self.shared);
        let gl = &shared.context.lock();
        for cmd_buf in command_buffers.iter() {
            // The command encoder assumes a default state when encoding the command buffer.
            // Always reset the state between command_buffers to reflect this assumption. Do
            // this at the beginning of the loop in case something outside of wgpu modified
            // this state prior to commit.
            unsafe { self.reset_state(gl) };
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(ref label) = cmd_buf.label {
                unsafe { gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, DEBUG_ID, label) };
            }

            for command in cmd_buf.commands.iter() {
                unsafe { self.process(gl, command, &cmd_buf.data_bytes, &cmd_buf.queries) };
            }

            #[cfg(not(target_arch = "wasm32"))]
            if cmd_buf.label.is_some() {
                unsafe { gl.pop_debug_group() };
            }
        }

        if let Some((fence, value)) = signal_fence {
            fence.maintain(gl);
            let sync = unsafe { gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) }
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
        #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
        let gl = unsafe { &self.shared.context.get_without_egl_lock() };

        #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
        let gl = &self.shared.context.glow_context;

        unsafe { surface.present(texture, gl) }
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        1.0
    }
}

// SAFE: WASM doesn't have threads
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for super::Queue {}
#[cfg(target_arch = "wasm32")]
unsafe impl Send for super::Queue {}
