use super::{conv, HResult as _, Resource};
use std::{mem, ops::Range};
use winapi::um::d3d12;

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        let list = match self.free_lists.pop() {
            Some(list) => {
                list.reset(self.allocator, native::PipelineState::null());
                list
            }
            None => self
                .device
                .create_graphics_command_list(
                    native::CmdListType::Direct,
                    self.allocator,
                    native::PipelineState::null(),
                    0,
                )
                .to_device_result("Create command list")?,
        };

        if let Some(label) = label {
            let cwstr = conv::map_label(label);
            list.SetName(cwstr.as_ptr());
        }
        self.list = Some(list);
        self.temp.clear();
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        if let Some(list) = self.list.take() {
            list.close();
            self.free_lists.push(list);
        }
    }
    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        let raw = self.list.take().unwrap();
        raw.close();
        Ok(super::CommandBuffer { raw })
    }
    unsafe fn reset_all<I: Iterator<Item = super::CommandBuffer>>(&mut self, command_buffers: I) {
        for cmd_buf in command_buffers {
            self.free_lists.push(cmd_buf.raw);
        }
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_buffer_usage_to_state(barrier.usage.start);
            let s1 = conv::map_buffer_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: barrier.buffer.resource.as_mut_ptr(),
                    Subresource: d3d12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: s0,
                    StateAfter: s1,
                };
                self.temp.barriers.push(raw);
            } else if barrier.usage.start == crate::BufferUses::STORAGE_STORE {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.UAV_mut() = d3d12::D3D12_RESOURCE_UAV_BARRIER {
                    pResource: barrier.buffer.resource.as_mut_ptr(),
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            self.list
                .unwrap()
                .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_texture_usage_to_state(barrier.usage.start);
            let s1 = conv::map_texture_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.Transition_mut() = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: barrier.texture.resource.as_mut_ptr(),
                    Subresource: d3d12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: s0,
                    StateAfter: s1,
                };

                let mip_level_count = match barrier.range.mip_level_count {
                    Some(count) => count.get(),
                    None => barrier.texture.mip_level_count - barrier.range.base_mip_level,
                };
                let array_layer_count = match barrier.range.array_layer_count {
                    Some(count) => count.get(),
                    None => barrier.texture.array_layer_count() - barrier.range.base_array_layer,
                };

                if barrier.range.aspect == wgt::TextureAspect::All
                    && barrier.range.base_mip_level + mip_level_count
                        == barrier.texture.mip_level_count
                    && barrier.range.base_array_layer + array_layer_count
                        == barrier.texture.array_layer_count()
                {
                    // Only one barrier if it affects the whole image.
                    self.temp.barriers.push(raw);
                } else {
                    // Generate barrier for each layer/level combination.
                    for rel_mip_level in 0..mip_level_count {
                        for rel_array_layer in 0..array_layer_count {
                            raw.u.Transition_mut().Subresource = barrier.texture.calc_subresource(
                                barrier.range.base_mip_level + rel_mip_level,
                                barrier.range.base_array_layer + rel_array_layer,
                                0,
                            );
                        }
                    }
                    self.temp.barriers.push(raw);
                }
            } else if barrier.usage.start == crate::TextureUses::STORAGE_STORE {
                let mut raw = d3d12::D3D12_RESOURCE_BARRIER {
                    Type: d3d12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: mem::zeroed(),
                };
                *raw.u.UAV_mut() = d3d12::D3D12_RESOURCE_UAV_BARRIER {
                    pResource: barrier.texture.resource.as_mut_ptr(),
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            self.list
                .unwrap()
                .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr());
        }
    }

    unsafe fn fill_buffer(
        &mut self,
        _buffer: &super::Buffer,
        _range: crate::MemoryRange,
        _value: u8,
    ) {
        //TODO
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let list = self.list.unwrap();
        for r in regions {
            list.CopyBufferRegion(
                dst.resource.as_mut_ptr(),
                r.dst_offset,
                src.resource.as_mut_ptr(),
                r.src_offset,
                r.size.get(),
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
        let list = self.list.unwrap();
        let mut src_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };
        let mut dst_location = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: mem::zeroed(),
        };

        for r in regions {
            let src_box = d3d12::D3D12_BOX {
                left: r.src_base.origin.x,
                top: r.src_base.origin.y,
                right: r.src_base.origin.x + r.size.width,
                bottom: r.src_base.origin.y + r.size.height,
                front: r.src_base.origin.z,
                back: r.src_base.origin.z + r.size.depth,
            };
            *src_location.u.SubresourceIndex_mut() =
                src.calc_subresource(r.src_base.mip_level, r.src_base.array_layer, 0);
            *dst_location.u.SubresourceIndex_mut() =
                dst.calc_subresource(r.dst_base.mip_level, r.dst_base.array_layer, 0);
            list.CopyTextureRegion(
                &dst_location,
                r.dst_base.origin.x,
                r.dst_base.origin.y,
                r.dst_base.origin.z,
                &src_location,
                &src_box,
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
        for _r in regions {}
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {}
    unsafe fn end_query(&mut self, set: &super::QuerySet, index: u32) {}
    unsafe fn write_timestamp(&mut self, set: &super::QuerySet, index: u32) {}
    unsafe fn reset_queries(&mut self, set: &super::QuerySet, range: Range<u32>) {}
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {}
    unsafe fn end_render_pass(&mut self) {}

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
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u32],
    ) {
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {}
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {}
    unsafe fn end_debug_marker(&mut self) {}

    unsafe fn set_render_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
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
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {}
    unsafe fn end_compute_pass(&mut self) {}

    unsafe fn set_compute_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {}
}
