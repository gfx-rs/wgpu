use super::Command as C;
use glow::HasContext;
use std::ops::Range;

const DEBUG_ID: u32 = 0;

fn extract_marker<'a>(data: &'a [u8], range: &Range<u32>) -> &'a str {
    std::str::from_utf8(&data[range.start as usize..range.end as usize]).unwrap()
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
                //TODO: bind src to self.copy_fbo
            }
            C::CopyBufferToTexture {
                src,
                src_target,
                dst,
                dst_target,
                ref copy,
            } => {
                //TODO: bind src to self.copy_fbo
            }
            C::CopyTextureToBuffer {
                src,
                src_target,
                dst,
                dst_target,
                ref copy,
            } => {
                //TODO: bind src to self.copy_fbo
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
