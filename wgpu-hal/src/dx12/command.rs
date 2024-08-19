use std::{mem, ops::Range};

use windows::Win32::{Foundation, Graphics::Direct3D12};

use super::conv;
use crate::{
    auxil::{self, dxgi::result::HResult as _},
    dx12::borrow_interface_temporarily,
};

fn make_box(origin: &wgt::Origin3d, size: &crate::CopyExtent) -> Direct3D12::D3D12_BOX {
    Direct3D12::D3D12_BOX {
        left: origin.x,
        top: origin.y,
        right: origin.x + size.width,
        bottom: origin.y + size.height,
        front: origin.z,
        back: origin.z + size.depth,
    }
}

impl crate::BufferTextureCopy {
    fn to_subresource_footprint(
        &self,
        format: wgt::TextureFormat,
    ) -> Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        let (block_width, _) = format.block_dimensions();
        Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: self.buffer_layout.offset,
            Footprint: Direct3D12::D3D12_SUBRESOURCE_FOOTPRINT {
                Format: auxil::dxgi::conv::map_texture_format_for_copy(
                    format,
                    self.texture_base.aspect,
                )
                .unwrap(),
                Width: self.size.width,
                Height: self.size.height,
                Depth: self.size.depth,
                RowPitch: {
                    let actual = self.buffer_layout.bytes_per_row.unwrap_or_else(|| {
                        // this may happen for single-line updates
                        let block_size = format
                            .block_copy_size(Some(self.texture_base.aspect.map()))
                            .unwrap();
                        (self.size.width / block_width) * block_size
                    });
                    wgt::math::align_to(actual, Direct3D12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
                },
            },
        }
    }
}

impl super::Temp {
    fn prepare_marker(&mut self, marker: &str) -> (&[u16], u32) {
        // TODO: Store in HSTRING
        self.marker.clear();
        self.marker.extend(marker.encode_utf16());
        self.marker.push(0);
        (&self.marker, self.marker.len() as u32 * 2)
    }
}

impl Drop for super::CommandEncoder {
    fn drop(&mut self) {
        use crate::CommandEncoder;
        unsafe { self.discard_encoding() }
    }
}

impl super::CommandEncoder {
    unsafe fn begin_pass(&mut self, kind: super::PassKind, label: crate::Label) {
        let list = self.list.as_ref().unwrap();
        self.pass.kind = kind;
        if let Some(label) = label {
            let (wide_label, size) = self.temp.prepare_marker(label);
            unsafe { list.BeginEvent(0, Some(wide_label.as_ptr().cast()), size) };
            self.pass.has_label = true;
        }
        self.pass.dirty_root_elements = 0;
        self.pass.dirty_vertex_buffers = 0;
        unsafe {
            list.SetDescriptorHeaps(&[
                Some(self.shared.heap_views.raw.clone()),
                Some(self.shared.heap_samplers.raw.clone()),
            ])
        };
    }

    unsafe fn end_pass(&mut self) {
        let list = self.list.as_ref().unwrap();
        unsafe { list.SetDescriptorHeaps(&[]) };
        if self.pass.has_label {
            unsafe { list.EndEvent() };
        }
        self.pass.clear();
    }

    unsafe fn prepare_draw(&mut self, first_vertex: i32, first_instance: u32) {
        while self.pass.dirty_vertex_buffers != 0 {
            let list = self.list.as_ref().unwrap();
            let index = self.pass.dirty_vertex_buffers.trailing_zeros();
            self.pass.dirty_vertex_buffers ^= 1 << index;
            unsafe {
                list.IASetVertexBuffers(
                    index,
                    Some(&self.pass.vertex_buffers[index as usize..][..1]),
                );
            }
        }
        if let Some(root_index) = self.pass.layout.special_constants_root_index {
            let needs_update = match self.pass.root_elements[root_index as usize] {
                super::RootElement::SpecialConstantBuffer {
                    first_vertex: other_vertex,
                    first_instance: other_instance,
                    other: _,
                } => first_vertex != other_vertex || first_instance != other_instance,
                _ => true,
            };
            if needs_update {
                self.pass.dirty_root_elements |= 1 << root_index;
                self.pass.root_elements[root_index as usize] =
                    super::RootElement::SpecialConstantBuffer {
                        first_vertex,
                        first_instance,
                        other: 0,
                    };
            }
        }
        self.update_root_elements();
    }

    fn prepare_dispatch(&mut self, count: [u32; 3]) {
        if let Some(root_index) = self.pass.layout.special_constants_root_index {
            let needs_update = match self.pass.root_elements[root_index as usize] {
                super::RootElement::SpecialConstantBuffer {
                    first_vertex,
                    first_instance,
                    other,
                } => [first_vertex as u32, first_instance, other] != count,
                _ => true,
            };
            if needs_update {
                self.pass.dirty_root_elements |= 1 << root_index;
                self.pass.root_elements[root_index as usize] =
                    super::RootElement::SpecialConstantBuffer {
                        first_vertex: count[0] as i32,
                        first_instance: count[1],
                        other: count[2],
                    };
            }
        }
        self.update_root_elements();
    }

    //Note: we have to call this lazily before draw calls. Otherwise, D3D complains
    // about the root parameters being incompatible with root signature.
    fn update_root_elements(&mut self) {
        use super::{BufferViewKind as Bvk, PassKind as Pk};

        while self.pass.dirty_root_elements != 0 {
            let list = self.list.as_ref().unwrap();
            let index = self.pass.dirty_root_elements.trailing_zeros();
            self.pass.dirty_root_elements ^= 1 << index;

            match self.pass.root_elements[index as usize] {
                super::RootElement::Empty => log::error!("Root index {} is not bound", index),
                super::RootElement::Constant => {
                    let info = self.pass.layout.root_constant_info.as_ref().unwrap();

                    for offset in info.range.clone() {
                        let val = self.pass.constant_data[offset as usize];
                        match self.pass.kind {
                            Pk::Render => unsafe {
                                list.SetGraphicsRoot32BitConstant(index, val, offset)
                            },
                            Pk::Compute => unsafe {
                                list.SetComputeRoot32BitConstant(index, val, offset)
                            },
                            Pk::Transfer => (),
                        }
                    }
                }
                super::RootElement::SpecialConstantBuffer {
                    first_vertex,
                    first_instance,
                    other,
                } => match self.pass.kind {
                    Pk::Render => {
                        unsafe { list.SetGraphicsRoot32BitConstant(index, first_vertex as u32, 0) };
                        unsafe { list.SetGraphicsRoot32BitConstant(index, first_instance, 1) };
                    }
                    Pk::Compute => {
                        unsafe { list.SetComputeRoot32BitConstant(index, first_vertex as u32, 0) };
                        unsafe { list.SetComputeRoot32BitConstant(index, first_instance, 1) };
                        unsafe { list.SetComputeRoot32BitConstant(index, other, 2) };
                    }
                    Pk::Transfer => (),
                },
                super::RootElement::Table(descriptor) => match self.pass.kind {
                    Pk::Render => unsafe { list.SetGraphicsRootDescriptorTable(index, descriptor) },
                    Pk::Compute => unsafe { list.SetComputeRootDescriptorTable(index, descriptor) },
                    Pk::Transfer => (),
                },
                super::RootElement::DynamicOffsetBuffer { kind, address } => {
                    let address = address.ptr;
                    match (self.pass.kind, kind) {
                        (Pk::Render, Bvk::Constant) => unsafe {
                            list.SetGraphicsRootConstantBufferView(index, address)
                        },
                        (Pk::Compute, Bvk::Constant) => unsafe {
                            list.SetComputeRootConstantBufferView(index, address)
                        },
                        (Pk::Render, Bvk::ShaderResource) => unsafe {
                            list.SetGraphicsRootShaderResourceView(index, address)
                        },
                        (Pk::Compute, Bvk::ShaderResource) => unsafe {
                            list.SetComputeRootShaderResourceView(index, address)
                        },
                        (Pk::Render, Bvk::UnorderedAccess) => unsafe {
                            list.SetGraphicsRootUnorderedAccessView(index, address)
                        },
                        (Pk::Compute, Bvk::UnorderedAccess) => unsafe {
                            list.SetComputeRootUnorderedAccessView(index, address)
                        },
                        (Pk::Transfer, _) => (),
                    }
                }
            }
        }
    }

    fn reset_signature(&mut self, layout: &super::PipelineLayoutShared) {
        if let Some(root_index) = layout.special_constants_root_index {
            self.pass.root_elements[root_index as usize] =
                super::RootElement::SpecialConstantBuffer {
                    first_vertex: 0,
                    first_instance: 0,
                    other: 0,
                };
        }
        self.pass.layout = layout.clone();
        self.pass.dirty_root_elements = (1 << layout.total_root_elements) - 1;
    }

    fn write_pass_end_timestamp_if_requested(&mut self) {
        if let Some((query_set_raw, index)) = self.end_of_pass_timer_query.take() {
            use crate::CommandEncoder as _;
            unsafe {
                self.write_timestamp(
                    &crate::dx12::QuerySet {
                        raw: query_set_raw,
                        raw_ty: Direct3D12::D3D12_QUERY_TYPE_TIMESTAMP,
                    },
                    index,
                );
            }
        }
    }
}

impl crate::CommandEncoder for super::CommandEncoder {
    type A = super::Api;

    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        let list = loop {
            if let Some(list) = self.free_lists.pop() {
                let reset_result = unsafe { list.Reset(&self.allocator, None) }.into_result();
                if reset_result.is_ok() {
                    break Some(list);
                }
            } else {
                break None;
            }
        };

        let list = if let Some(list) = list {
            list
        } else {
            unsafe {
                self.device.CreateCommandList(
                    0,
                    Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &self.allocator,
                    None,
                )
            }
            .into_device_result("Create command list")?
        };

        if let Some(label) = label {
            unsafe { list.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.list = Some(list);
        self.temp.clear();
        self.pass.clear();
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        if let Some(list) = self.list.take() {
            if unsafe { list.Close() }.is_ok() {
                self.free_lists.push(list);
            }
        }
    }
    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        let raw = self.list.take().unwrap();
        unsafe { raw.Close() }.into_device_result("GraphicsCommandList::close")?;
        Ok(super::CommandBuffer { raw })
    }
    unsafe fn reset_all<I: Iterator<Item = super::CommandBuffer>>(&mut self, command_buffers: I) {
        for cmd_buf in command_buffers {
            self.free_lists.push(cmd_buf.raw);
        }
        let _todo_handle_error = unsafe { self.allocator.Reset() };
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Buffer>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_buffer_usage_to_state(barrier.usage.start);
            let s1 = conv::map_buffer_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let raw = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        Transition: mem::ManuallyDrop::new(
                            Direct3D12::D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe {
                                    borrow_interface_temporarily(&barrier.buffer.resource)
                                },
                                Subresource: Direct3D12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: s0,
                                StateAfter: s1,
                            },
                        ),
                    },
                };
                self.temp.barriers.push(raw);
            } else if barrier.usage.start == crate::BufferUses::STORAGE_READ_WRITE {
                let raw = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        UAV: mem::ManuallyDrop::new(Direct3D12::D3D12_RESOURCE_UAV_BARRIER {
                            pResource: unsafe {
                                borrow_interface_temporarily(&barrier.buffer.resource)
                            },
                        }),
                    },
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            unsafe {
                self.list
                    .as_ref()
                    .unwrap()
                    .ResourceBarrier(&self.temp.barriers)
            };
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Texture>>,
    {
        self.temp.barriers.clear();

        for barrier in barriers {
            let s0 = conv::map_texture_usage_to_state(barrier.usage.start);
            let s1 = conv::map_texture_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        Transition: mem::ManuallyDrop::new(
                            Direct3D12::D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe {
                                    borrow_interface_temporarily(&barrier.texture.resource)
                                },
                                Subresource: Direct3D12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: s0,
                                StateAfter: s1,
                            },
                        ),
                    },
                };

                let tex_mip_level_count = barrier.texture.mip_level_count;
                let tex_array_layer_count = barrier.texture.array_layer_count();

                if barrier.range.is_full_resource(
                    barrier.texture.format,
                    tex_mip_level_count,
                    tex_array_layer_count,
                ) {
                    // Only one barrier if it affects the whole image.
                    self.temp.barriers.push(raw);
                } else {
                    // Selected texture aspect is relevant if the texture format has both depth _and_ stencil aspects.
                    let planes = if barrier.texture.format.is_combined_depth_stencil_format() {
                        match barrier.range.aspect {
                            wgt::TextureAspect::All => 0..2,
                            wgt::TextureAspect::DepthOnly => 0..1,
                            wgt::TextureAspect::StencilOnly => 1..2,
                            _ => unreachable!(),
                        }
                    } else if let Some(planes) = barrier.texture.format.planes() {
                        match barrier.range.aspect {
                            wgt::TextureAspect::All => 0..planes,
                            wgt::TextureAspect::Plane0 => 0..1,
                            wgt::TextureAspect::Plane1 => 1..2,
                            wgt::TextureAspect::Plane2 => 2..3,
                            _ => unreachable!(),
                        }
                    } else {
                        match barrier.texture.format {
                            wgt::TextureFormat::Stencil8 => 1..2,
                            wgt::TextureFormat::Depth24Plus => 0..2, // TODO: investigate why tests fail if we set this to 0..1
                            _ => 0..1,
                        }
                    };

                    for mip_level in barrier.range.mip_range(tex_mip_level_count) {
                        for array_layer in barrier.range.layer_range(tex_array_layer_count) {
                            for plane in planes.clone() {
                                unsafe { &mut *raw.Anonymous.Transition }.Subresource = barrier
                                    .texture
                                    .calc_subresource(mip_level, array_layer, plane);
                                self.temp.barriers.push(raw.clone());
                            }
                        }
                    }
                }
            } else if barrier.usage.start == crate::TextureUses::STORAGE_READ_WRITE {
                let raw = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        UAV: mem::ManuallyDrop::new(Direct3D12::D3D12_RESOURCE_UAV_BARRIER {
                            pResource: unsafe {
                                borrow_interface_temporarily(&barrier.texture.resource)
                            },
                        }),
                    },
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            unsafe {
                self.list
                    .as_ref()
                    .unwrap()
                    .ResourceBarrier(&self.temp.barriers)
            };
        }
    }

    unsafe fn clear_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange) {
        let list = self.list.as_ref().unwrap();
        let mut offset = range.start;
        while offset < range.end {
            let size = super::ZERO_BUFFER_SIZE.min(range.end - offset);
            unsafe {
                list.CopyBufferRegion(&buffer.resource, offset, &self.shared.zero_buffer, 0, size)
            };
            offset += size;
        }
    }

    unsafe fn copy_buffer_to_buffer<T>(
        &mut self,
        src: &super::Buffer,
        dst: &super::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = crate::BufferCopy>,
    {
        let list = self.list.as_ref().unwrap();
        for r in regions {
            unsafe {
                list.CopyBufferRegion(
                    &dst.resource,
                    r.dst_offset,
                    &src.resource,
                    r.src_offset,
                    r.size.get(),
                )
            };
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
        let list = self.list.as_ref().unwrap();

        for r in regions {
            let src_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&src.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: src.calc_subresource_for_copy(&r.src_base),
                },
            };
            let dst_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&dst.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: dst.calc_subresource_for_copy(&r.dst_base),
                },
            };

            let src_box = make_box(&r.src_base.origin, &r.size);

            unsafe {
                list.CopyTextureRegion(
                    &dst_location,
                    r.dst_base.origin.x,
                    r.dst_base.origin.y,
                    r.dst_base.origin.z,
                    &src_location,
                    Some(&src_box),
                )
            };
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
        let list = self.list.as_ref().unwrap();
        for r in regions {
            let src_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&src.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: r.to_subresource_footprint(dst.format),
                },
            };
            let dst_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&dst.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: dst.calc_subresource_for_copy(&r.texture_base),
                },
            };

            let src_box = make_box(&wgt::Origin3d::ZERO, &r.size);
            unsafe {
                list.CopyTextureRegion(
                    &dst_location,
                    r.texture_base.origin.x,
                    r.texture_base.origin.y,
                    r.texture_base.origin.z,
                    &src_location,
                    Some(&src_box),
                )
            };
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
        let list = self.list.as_ref().unwrap();
        for r in regions {
            let src_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&src.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: src.calc_subresource_for_copy(&r.texture_base),
                },
            };
            let dst_location = Direct3D12::D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { borrow_interface_temporarily(&dst.resource) },
                Type: Direct3D12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: Direct3D12::D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: r.to_subresource_footprint(src.format),
                },
            };

            let src_box = make_box(&r.texture_base.origin, &r.size);
            unsafe {
                list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, Some(&src_box))
            };
        }
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .BeginQuery(&set.raw, set.raw_ty, index)
        };
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .EndQuery(&set.raw, set.raw_ty, index)
        };
    }
    unsafe fn write_timestamp(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list.as_ref().unwrap().EndQuery(
                &set.raw,
                Direct3D12::D3D12_QUERY_TYPE_TIMESTAMP,
                index,
            )
        };
    }
    unsafe fn reset_queries(&mut self, _set: &super::QuerySet, _range: Range<u32>) {
        // nothing to do here
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &super::QuerySet,
        range: Range<u32>,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        _stride: wgt::BufferSize,
    ) {
        unsafe {
            self.list.as_ref().unwrap().ResolveQueryData(
                &set.raw,
                set.raw_ty,
                range.start,
                range.end - range.start,
                &buffer.resource,
                offset,
            )
        };
    }

    // render

    unsafe fn begin_render_pass(
        &mut self,
        desc: &crate::RenderPassDescriptor<super::QuerySet, super::TextureView>,
    ) {
        unsafe { self.begin_pass(super::PassKind::Render, desc.label) };

        // Start timestamp if any (before all other commands but after debug marker)
        if let Some(timestamp_writes) = desc.timestamp_writes.as_ref() {
            if let Some(index) = timestamp_writes.beginning_of_pass_write_index {
                unsafe {
                    self.write_timestamp(timestamp_writes.query_set, index);
                }
            }
            self.end_of_pass_timer_query = timestamp_writes
                .end_of_pass_write_index
                .map(|index| (timestamp_writes.query_set.raw.clone(), index));
        }

        let mut color_views =
            [Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }; crate::MAX_COLOR_ATTACHMENTS];
        for (rtv, cat) in color_views.iter_mut().zip(desc.color_attachments.iter()) {
            if let Some(cat) = cat.as_ref() {
                *rtv = cat.target.view.handle_rtv.unwrap().raw;
            } else {
                *rtv = self.null_rtv_handle.raw;
            }
        }

        let ds_view = desc.depth_stencil_attachment.as_ref().map(|ds| {
            if ds.target.usage == crate::TextureUses::DEPTH_STENCIL_WRITE {
                ds.target.view.handle_dsv_rw.as_ref().unwrap().raw
            } else {
                ds.target.view.handle_dsv_ro.as_ref().unwrap().raw
            }
        });

        let list = self.list.as_ref().unwrap();
        #[allow(trivial_casts)] // No other clean way to write the coercion inside .map() below?
        unsafe {
            list.OMSetRenderTargets(
                desc.color_attachments.len() as u32,
                Some(color_views.as_ptr()),
                false,
                ds_view.as_ref().map(std::ptr::from_ref),
            )
        };

        self.pass.resolves.clear();
        for (rtv, cat) in color_views.iter().zip(desc.color_attachments.iter()) {
            if let Some(cat) = cat.as_ref() {
                if !cat.ops.contains(crate::AttachmentOps::LOAD) {
                    let value = [
                        cat.clear_value.r as f32,
                        cat.clear_value.g as f32,
                        cat.clear_value.b as f32,
                        cat.clear_value.a as f32,
                    ];
                    // TODO: Empty slice vs None?
                    unsafe { list.ClearRenderTargetView(*rtv, &value, Some(&[])) };
                }
                if let Some(ref target) = cat.resolve_target {
                    self.pass.resolves.push(super::PassResolve {
                        src: cat.target.view.target_base.clone(),
                        dst: target.view.target_base.clone(),
                        format: target.view.raw_format,
                    });
                }
            }
        }

        if let Some(ref ds) = desc.depth_stencil_attachment {
            let mut flags = Direct3D12::D3D12_CLEAR_FLAGS::default();
            let aspects = ds.target.view.aspects;
            if !ds.depth_ops.contains(crate::AttachmentOps::LOAD)
                && aspects.contains(crate::FormatAspects::DEPTH)
            {
                flags |= Direct3D12::D3D12_CLEAR_FLAG_DEPTH;
            }
            if !ds.stencil_ops.contains(crate::AttachmentOps::LOAD)
                && aspects.contains(crate::FormatAspects::STENCIL)
            {
                flags |= Direct3D12::D3D12_CLEAR_FLAG_STENCIL;
            }

            if let Some(ds_view) = ds_view {
                if flags != Direct3D12::D3D12_CLEAR_FLAGS::default() {
                    unsafe {
                        list.ClearDepthStencilView(
                            ds_view,
                            flags,
                            ds.clear_value.0,
                            ds.clear_value.1 as u8,
                            &[],
                        )
                    }
                }
            }
        }

        let raw_vp = Direct3D12::D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: desc.extent.width as f32,
            Height: desc.extent.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let raw_rect = Foundation::RECT {
            left: 0,
            top: 0,
            right: desc.extent.width as i32,
            bottom: desc.extent.height as i32,
        };
        unsafe { list.RSSetViewports(std::slice::from_ref(&raw_vp)) };
        unsafe { list.RSSetScissorRects(std::slice::from_ref(&raw_rect)) };
    }

    unsafe fn end_render_pass(&mut self) {
        if !self.pass.resolves.is_empty() {
            let list = self.list.as_ref().unwrap();
            self.temp.barriers.clear();

            // All the targets are expected to be in `COLOR_TARGET` state,
            // but D3D12 has special source/destination states for the resolves.
            for resolve in self.pass.resolves.iter() {
                let barrier = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        //Note: this assumes `D3D12_RESOURCE_STATE_RENDER_TARGET`.
                        // If it's not the case, we can include the `TextureUses` in `PassResove`.
                        Transition: mem::ManuallyDrop::new(
                            Direct3D12::D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe { borrow_interface_temporarily(&resolve.src.0) },
                                Subresource: resolve.src.1,
                                StateBefore: Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET,
                                StateAfter: Direct3D12::D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                            },
                        ),
                    },
                };
                self.temp.barriers.push(barrier);
                let barrier = Direct3D12::D3D12_RESOURCE_BARRIER {
                    Type: Direct3D12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Direct3D12::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: Direct3D12::D3D12_RESOURCE_BARRIER_0 {
                        //Note: this assumes `D3D12_RESOURCE_STATE_RENDER_TARGET`.
                        // If it's not the case, we can include the `TextureUses` in `PassResolve`.
                        Transition: mem::ManuallyDrop::new(
                            Direct3D12::D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: unsafe { borrow_interface_temporarily(&resolve.dst.0) },
                                Subresource: resolve.dst.1,
                                StateBefore: Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET,
                                StateAfter: Direct3D12::D3D12_RESOURCE_STATE_RESOLVE_DEST,
                            },
                        ),
                    },
                };
                self.temp.barriers.push(barrier);
            }

            if !self.temp.barriers.is_empty() {
                profiling::scope!("ID3D12GraphicsCommandList::ResourceBarrier");
                unsafe { list.ResourceBarrier(&self.temp.barriers) };
            }

            for resolve in self.pass.resolves.iter() {
                profiling::scope!("ID3D12GraphicsCommandList::ResolveSubresource");
                unsafe {
                    list.ResolveSubresource(
                        &resolve.dst.0,
                        resolve.dst.1,
                        &resolve.src.0,
                        resolve.src.1,
                        resolve.format,
                    )
                };
            }

            // Flip all the barriers to reverse, back into `COLOR_TARGET`.
            for barrier in self.temp.barriers.iter_mut() {
                let transition = unsafe { &mut *barrier.Anonymous.Transition };
                mem::swap(&mut transition.StateBefore, &mut transition.StateAfter);
            }
            if !self.temp.barriers.is_empty() {
                profiling::scope!("ID3D12GraphicsCommandList::ResourceBarrier");
                unsafe { list.ResourceBarrier(&self.temp.barriers) };
            }
        }

        self.write_pass_end_timestamp_if_requested();

        unsafe { self.end_pass() };
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &super::PipelineLayout,
        index: u32,
        group: &super::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        let info = &layout.bind_group_infos[index as usize];
        let mut root_index = info.base_root_index as usize;

        // Bind CBV/SRC/UAV descriptor tables
        if info.tables.contains(super::TableTypes::SRV_CBV_UAV) {
            self.pass.root_elements[root_index] =
                super::RootElement::Table(group.handle_views.unwrap().gpu);
            root_index += 1;
        }

        // Bind Sampler descriptor tables.
        if info.tables.contains(super::TableTypes::SAMPLERS) {
            self.pass.root_elements[root_index] =
                super::RootElement::Table(group.handle_samplers.unwrap().gpu);
            root_index += 1;
        }

        // Bind root descriptors
        for ((&kind, &gpu_base), &offset) in info
            .dynamic_buffers
            .iter()
            .zip(group.dynamic_buffers.iter())
            .zip(dynamic_offsets)
        {
            self.pass.root_elements[root_index] = super::RootElement::DynamicOffsetBuffer {
                kind,
                address: Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_base.ptr + offset as u64,
                },
            };
            root_index += 1;
        }

        if self.pass.layout.signature == layout.shared.signature {
            self.pass.dirty_root_elements |= (1 << root_index) - (1 << info.base_root_index);
        } else {
            // D3D12 requires full reset on signature change
            self.reset_signature(&layout.shared);
        };
    }
    unsafe fn set_push_constants(
        &mut self,
        layout: &super::PipelineLayout,
        _stages: wgt::ShaderStages,
        offset_bytes: u32,
        data: &[u32],
    ) {
        let offset_words = offset_bytes as usize / 4;

        let info = layout.shared.root_constant_info.as_ref().unwrap();

        self.pass.root_elements[info.root_index as usize] = super::RootElement::Constant;

        self.pass.constant_data[offset_words..(offset_words + data.len())].copy_from_slice(data);

        if self.pass.layout.signature == layout.shared.signature {
            self.pass.dirty_root_elements |= 1 << info.root_index;
        } else {
            // D3D12 requires full reset on signature change
            self.reset_signature(&layout.shared);
        };
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        let (wide_label, size) = self.temp.prepare_marker(label);
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .SetMarker(0, Some(wide_label.as_ptr().cast()), size)
        };
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        let (wide_label, size) = self.temp.prepare_marker(group_label);
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .BeginEvent(0, Some(wide_label.as_ptr().cast()), size)
        };
    }
    unsafe fn end_debug_marker(&mut self) {
        unsafe { self.list.as_ref().unwrap().EndEvent() }
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        let list = self.list.clone().unwrap();

        if self.pass.layout.signature != pipeline.layout.signature {
            // D3D12 requires full reset on signature change
            unsafe { list.SetGraphicsRootSignature(pipeline.layout.signature.as_ref()) };
            self.reset_signature(&pipeline.layout);
        };

        unsafe { list.SetPipelineState(&pipeline.raw) };
        unsafe { list.IASetPrimitiveTopology(pipeline.topology) };

        for (index, (vb, &stride)) in self
            .pass
            .vertex_buffers
            .iter_mut()
            .zip(pipeline.vertex_strides.iter())
            .enumerate()
        {
            if let Some(stride) = stride {
                if vb.StrideInBytes != stride.get() {
                    vb.StrideInBytes = stride.get();
                    self.pass.dirty_vertex_buffers |= 1 << index;
                }
            }
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, super::Buffer>,
        format: wgt::IndexFormat,
    ) {
        let ibv = Direct3D12::D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: binding.resolve_address(),
            SizeInBytes: binding.resolve_size() as u32,
            Format: auxil::dxgi::conv::map_index_format(format),
        };

        unsafe { self.list.as_ref().unwrap().IASetIndexBuffer(Some(&ibv)) }
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Buffer>,
    ) {
        let vb = &mut self.pass.vertex_buffers[index as usize];
        vb.BufferLocation = binding.resolve_address();
        vb.SizeInBytes = binding.resolve_size() as u32;
        self.pass.dirty_vertex_buffers |= 1 << index;
    }

    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let raw_vp = Direct3D12::D3D12_VIEWPORT {
            TopLeftX: rect.x,
            TopLeftY: rect.y,
            Width: rect.w,
            Height: rect.h,
            MinDepth: depth_range.start,
            MaxDepth: depth_range.end,
        };
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .RSSetViewports(std::slice::from_ref(&raw_vp))
        }
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        let raw_rect = Foundation::RECT {
            left: rect.x as i32,
            top: rect.y as i32,
            right: (rect.x + rect.w) as i32,
            bottom: (rect.y + rect.h) as i32,
        };
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .RSSetScissorRects(std::slice::from_ref(&raw_rect))
        }
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        unsafe { self.list.as_ref().unwrap().OMSetStencilRef(value) }
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        unsafe { self.list.as_ref().unwrap().OMSetBlendFactor(Some(color)) }
    }

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe { self.prepare_draw(first_vertex as i32, first_instance) };
        unsafe {
            self.list.as_ref().unwrap().DrawInstanced(
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        }
    }
    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe { self.prepare_draw(base_vertex, first_instance) };
        unsafe {
            self.list.as_ref().unwrap().DrawIndexedInstanced(
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            )
        }
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        unsafe { self.prepare_draw(0, 0) };
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                &self.shared.cmd_signatures.draw,
                draw_count,
                &buffer.resource,
                offset,
                None,
                0,
            )
        }
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        unsafe { self.prepare_draw(0, 0) };
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                &self.shared.cmd_signatures.draw_indexed,
                draw_count,
                &buffer.resource,
                offset,
                None,
                0,
            )
        }
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        unsafe { self.prepare_draw(0, 0) };
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                &self.shared.cmd_signatures.draw,
                max_count,
                &buffer.resource,
                offset,
                &count_buffer.resource,
                count_offset,
            )
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
        unsafe { self.prepare_draw(0, 0) };
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                &self.shared.cmd_signatures.draw_indexed,
                max_count,
                &buffer.resource,
                offset,
                &count_buffer.resource,
                count_offset,
            )
        }
    }

    // compute

    unsafe fn begin_compute_pass<'a>(
        &mut self,
        desc: &crate::ComputePassDescriptor<'a, super::QuerySet>,
    ) {
        unsafe { self.begin_pass(super::PassKind::Compute, desc.label) };

        if let Some(timestamp_writes) = desc.timestamp_writes.as_ref() {
            if let Some(index) = timestamp_writes.beginning_of_pass_write_index {
                unsafe {
                    self.write_timestamp(timestamp_writes.query_set, index);
                }
            }
            self.end_of_pass_timer_query = timestamp_writes
                .end_of_pass_write_index
                .map(|index| (timestamp_writes.query_set.raw.clone(), index));
        }
    }
    unsafe fn end_compute_pass(&mut self) {
        self.write_pass_end_timestamp_if_requested();
        unsafe { self.end_pass() };
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &super::ComputePipeline) {
        let list = self.list.clone().unwrap();

        if self.pass.layout.signature != pipeline.layout.signature {
            // D3D12 requires full reset on signature change
            unsafe { list.SetComputeRootSignature(pipeline.layout.signature.as_ref()) };
            self.reset_signature(&pipeline.layout);
        };

        unsafe { list.SetPipelineState(&pipeline.raw) }
    }

    unsafe fn dispatch(&mut self, count @ [x, y, z]: [u32; 3]) {
        self.prepare_dispatch(count);
        unsafe { self.list.as_ref().unwrap().Dispatch(x, y, z) }
    }

    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.prepare_dispatch([0; 3]);
        //TODO: update special constants indirectly
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                &self.shared.cmd_signatures.dispatch,
                1,
                &buffer.resource,
                offset,
                None,
                0,
            )
        }
    }

    unsafe fn build_acceleration_structures<'a, T>(
        &mut self,
        _descriptor_count: u32,
        _descriptors: T,
    ) where
        super::Api: 'a,
        T: IntoIterator<
            Item = crate::BuildAccelerationStructureDescriptor<
                'a,
                super::Buffer,
                super::AccelerationStructure,
            >,
        >,
    {
        // Implement using `BuildRaytracingAccelerationStructure`:
        // https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#buildraytracingaccelerationstructure
        todo!()
    }

    unsafe fn place_acceleration_structure_barrier(
        &mut self,
        _barriers: crate::AccelerationStructureBarrier,
    ) {
        todo!()
    }
}
