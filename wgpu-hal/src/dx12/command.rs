use crate::auxil::{self, dxgi::result::HResult as _};

use super::conv;
use std::{mem, ops::Range, ptr};
use winapi::um::d3d12 as d3d12_ty;

fn make_box(origin: &wgt::Origin3d, size: &crate::CopyExtent) -> d3d12_ty::D3D12_BOX {
    d3d12_ty::D3D12_BOX {
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
    ) -> d3d12_ty::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        let (block_width, block_height) = format.block_dimensions();
        d3d12_ty::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: self.buffer_layout.offset,
            Footprint: d3d12_ty::D3D12_SUBRESOURCE_FOOTPRINT {
                Format: auxil::dxgi::conv::map_texture_format_for_copy(
                    format,
                    self.texture_base.aspect,
                )
                .unwrap(),
                Width: self.size.width,
                Height: self
                    .buffer_layout
                    .rows_per_image
                    .map_or(self.size.height, |count| count * block_height),
                Depth: self.size.depth,
                RowPitch: {
                    let actual = self.buffer_layout.bytes_per_row.unwrap_or_else(|| {
                        // this may happen for single-line updates
                        let block_size = format
                            .block_copy_size(Some(self.texture_base.aspect.map()))
                            .unwrap();
                        (self.size.width / block_width) * block_size
                    });
                    wgt::math::align_to(actual, d3d12_ty::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
                },
            },
        }
    }
}

impl super::Temp {
    fn prepare_marker(&mut self, marker: &str) -> (&[u16], u32) {
        self.marker.clear();
        self.marker.extend(marker.encode_utf16());
        self.marker.push(0);
        (&self.marker, self.marker.len() as u32 * 2)
    }
}

impl super::CommandEncoder {
    unsafe fn begin_pass(&mut self, kind: super::PassKind, label: crate::Label) {
        let list = self.list.as_ref().unwrap();
        self.pass.kind = kind;
        if let Some(label) = label {
            let (wide_label, size) = self.temp.prepare_marker(label);
            unsafe { list.BeginEvent(0, wide_label.as_ptr() as *const _, size) };
            self.pass.has_label = true;
        }
        self.pass.dirty_root_elements = 0;
        self.pass.dirty_vertex_buffers = 0;
        list.set_descriptor_heaps(&[
            self.shared.heap_views.raw.clone(),
            self.shared.heap_samplers.raw.clone(),
        ]);
    }

    unsafe fn end_pass(&mut self) {
        let list = self.list.as_ref().unwrap();
        list.set_descriptor_heaps(&[]);
        if self.pass.has_label {
            unsafe { list.EndEvent() };
        }
        self.pass.clear();
    }

    unsafe fn prepare_draw(&mut self, base_vertex: i32, base_instance: u32) {
        while self.pass.dirty_vertex_buffers != 0 {
            let list = self.list.as_ref().unwrap();
            let index = self.pass.dirty_vertex_buffers.trailing_zeros();
            self.pass.dirty_vertex_buffers ^= 1 << index;
            unsafe {
                list.IASetVertexBuffers(
                    index,
                    1,
                    self.pass.vertex_buffers.as_ptr().offset(index as isize),
                );
            }
        }
        if let Some(root_index) = self.pass.layout.special_constants_root_index {
            let needs_update = match self.pass.root_elements[root_index as usize] {
                super::RootElement::SpecialConstantBuffer {
                    base_vertex: other_vertex,
                    base_instance: other_instance,
                    other: _,
                } => base_vertex != other_vertex || base_instance != other_instance,
                _ => true,
            };
            if needs_update {
                self.pass.dirty_root_elements |= 1 << root_index;
                self.pass.root_elements[root_index as usize] =
                    super::RootElement::SpecialConstantBuffer {
                        base_vertex,
                        base_instance,
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
                    base_vertex,
                    base_instance,
                    other,
                } => [base_vertex as u32, base_instance, other] != count,
                _ => true,
            };
            if needs_update {
                self.pass.dirty_root_elements |= 1 << root_index;
                self.pass.root_elements[root_index as usize] =
                    super::RootElement::SpecialConstantBuffer {
                        base_vertex: count[0] as i32,
                        base_instance: count[1],
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
                            Pk::Render => list.set_graphics_root_constant(index, val, offset),
                            Pk::Compute => list.set_compute_root_constant(index, val, offset),
                            Pk::Transfer => (),
                        }
                    }
                }
                super::RootElement::SpecialConstantBuffer {
                    base_vertex,
                    base_instance,
                    other,
                } => match self.pass.kind {
                    Pk::Render => {
                        list.set_graphics_root_constant(index, base_vertex as u32, 0);
                        list.set_graphics_root_constant(index, base_instance, 1);
                    }
                    Pk::Compute => {
                        list.set_compute_root_constant(index, base_vertex as u32, 0);
                        list.set_compute_root_constant(index, base_instance, 1);
                        list.set_compute_root_constant(index, other, 2);
                    }
                    Pk::Transfer => (),
                },
                super::RootElement::Table(descriptor) => match self.pass.kind {
                    Pk::Render => list.set_graphics_root_descriptor_table(index, descriptor),
                    Pk::Compute => list.set_compute_root_descriptor_table(index, descriptor),
                    Pk::Transfer => (),
                },
                super::RootElement::DynamicOffsetBuffer { kind, address } => {
                    match (self.pass.kind, kind) {
                        (Pk::Render, Bvk::Constant) => {
                            list.set_graphics_root_constant_buffer_view(index, address)
                        }
                        (Pk::Compute, Bvk::Constant) => {
                            list.set_compute_root_constant_buffer_view(index, address)
                        }
                        (Pk::Render, Bvk::ShaderResource) => {
                            list.set_graphics_root_shader_resource_view(index, address)
                        }
                        (Pk::Compute, Bvk::ShaderResource) => {
                            list.set_compute_root_shader_resource_view(index, address)
                        }
                        (Pk::Render, Bvk::UnorderedAccess) => {
                            list.set_graphics_root_unordered_access_view(index, address)
                        }
                        (Pk::Compute, Bvk::UnorderedAccess) => {
                            list.set_compute_root_unordered_access_view(index, address)
                        }
                        (Pk::Transfer, _) => (),
                    }
                }
            }
        }
    }

    fn reset_signature(&mut self, layout: &super::PipelineLayoutShared) {
        log::trace!("Reset signature {:?}", layout.signature);
        if let Some(root_index) = layout.special_constants_root_index {
            self.pass.root_elements[root_index as usize] =
                super::RootElement::SpecialConstantBuffer {
                    base_vertex: 0,
                    base_instance: 0,
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
                        raw_ty: d3d12_ty::D3D12_QUERY_TYPE_TIMESTAMP,
                    },
                    index,
                );
            }
        }
    }
}

impl crate::CommandEncoder<super::Api> for super::CommandEncoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        let list = loop {
            if let Some(list) = self.free_lists.pop() {
                let reset_result = list
                    .reset(&self.allocator, d3d12::PipelineState::null())
                    .into_result();
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
            self.device
                .create_graphics_command_list(
                    d3d12::CmdListType::Direct,
                    &self.allocator,
                    d3d12::PipelineState::null(),
                    0,
                )
                .into_device_result("Create command list")?
        };

        if let Some(label) = label {
            let cwstr = conv::map_label(label);
            unsafe { list.SetName(cwstr.as_ptr()) };
        }

        self.list = Some(list);
        self.temp.clear();
        self.pass.clear();
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {
        if let Some(list) = self.list.take() {
            if list.close().into_result().is_ok() {
                self.free_lists.push(list);
            }
        }
    }
    unsafe fn end_encoding(&mut self) -> Result<super::CommandBuffer, crate::DeviceError> {
        let raw = self.list.take().unwrap();
        let closed = raw.close().into_result().is_ok();
        Ok(super::CommandBuffer { raw, closed })
    }
    unsafe fn reset_all<I: Iterator<Item = super::CommandBuffer>>(&mut self, command_buffers: I) {
        for cmd_buf in command_buffers {
            if cmd_buf.closed {
                self.free_lists.push(cmd_buf.raw);
            }
        }
        self.allocator.reset();
    }

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        log::trace!(
            "List {:p} buffer transitions",
            self.list.as_ref().unwrap().as_ptr()
        );
        for barrier in barriers {
            log::trace!(
                "\t{:p}: usage {:?}..{:?}",
                barrier.buffer.resource.as_ptr(),
                barrier.usage.start,
                barrier.usage.end
            );
            let s0 = conv::map_buffer_usage_to_state(barrier.usage.start);
            let s1 = conv::map_buffer_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12_ty::D3D12_RESOURCE_BARRIER {
                    Type: d3d12_ty::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12_ty::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: unsafe { mem::zeroed() },
                };
                unsafe {
                    *raw.u.Transition_mut() = d3d12_ty::D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: barrier.buffer.resource.as_mut_ptr(),
                        Subresource: d3d12_ty::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: s0,
                        StateAfter: s1,
                    }
                };
                self.temp.barriers.push(raw);
            } else if barrier.usage.start == crate::BufferUses::STORAGE_READ_WRITE {
                let mut raw = d3d12_ty::D3D12_RESOURCE_BARRIER {
                    Type: d3d12_ty::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12_ty::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: unsafe { mem::zeroed() },
                };
                unsafe {
                    *raw.u.UAV_mut() = d3d12_ty::D3D12_RESOURCE_UAV_BARRIER {
                        pResource: barrier.buffer.resource.as_mut_ptr(),
                    }
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            unsafe {
                self.list
                    .as_ref()
                    .unwrap()
                    .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr())
            };
        }
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, super::Api>>,
    {
        self.temp.barriers.clear();

        log::trace!(
            "List {:p} texture transitions",
            self.list.as_ref().unwrap().as_ptr()
        );
        for barrier in barriers {
            log::trace!(
                "\t{:p}: usage {:?}..{:?}, range {:?}",
                barrier.texture.resource.as_ptr(),
                barrier.usage.start,
                barrier.usage.end,
                barrier.range
            );
            let s0 = conv::map_texture_usage_to_state(barrier.usage.start);
            let s1 = conv::map_texture_usage_to_state(barrier.usage.end);
            if s0 != s1 {
                let mut raw = d3d12_ty::D3D12_RESOURCE_BARRIER {
                    Type: d3d12_ty::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12_ty::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: unsafe { mem::zeroed() },
                };
                unsafe {
                    *raw.u.Transition_mut() = d3d12_ty::D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: barrier.texture.resource.as_mut_ptr(),
                        Subresource: d3d12_ty::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: s0,
                        StateAfter: s1,
                    }
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
                                unsafe {
                                    raw.u.Transition_mut().Subresource = barrier
                                        .texture
                                        .calc_subresource(mip_level, array_layer, plane);
                                };
                                self.temp.barriers.push(raw);
                            }
                        }
                    }
                }
            } else if barrier.usage.start == crate::TextureUses::STORAGE_READ_WRITE {
                let mut raw = d3d12_ty::D3D12_RESOURCE_BARRIER {
                    Type: d3d12_ty::D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: d3d12_ty::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: unsafe { mem::zeroed() },
                };
                unsafe {
                    *raw.u.UAV_mut() = d3d12_ty::D3D12_RESOURCE_UAV_BARRIER {
                        pResource: barrier.texture.resource.as_mut_ptr(),
                    }
                };
                self.temp.barriers.push(raw);
            }
        }

        if !self.temp.barriers.is_empty() {
            unsafe {
                self.list
                    .as_ref()
                    .unwrap()
                    .ResourceBarrier(self.temp.barriers.len() as u32, self.temp.barriers.as_ptr())
            };
        }
    }

    unsafe fn clear_buffer(&mut self, buffer: &super::Buffer, range: crate::MemoryRange) {
        let list = self.list.as_ref().unwrap();
        let mut offset = range.start;
        while offset < range.end {
            let size = super::ZERO_BUFFER_SIZE.min(range.end - offset);
            unsafe {
                list.CopyBufferRegion(
                    buffer.resource.as_mut_ptr(),
                    offset,
                    self.shared.zero_buffer.as_mut_ptr(),
                    0,
                    size,
                )
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
                    dst.resource.as_mut_ptr(),
                    r.dst_offset,
                    src.resource.as_mut_ptr(),
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
        let mut src_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: unsafe { mem::zeroed() },
        };
        let mut dst_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: unsafe { mem::zeroed() },
        };

        for r in regions {
            let src_box = make_box(&r.src_base.origin, &r.size);
            unsafe {
                *src_location.u.SubresourceIndex_mut() = src.calc_subresource_for_copy(&r.src_base)
            };
            unsafe {
                *dst_location.u.SubresourceIndex_mut() = dst.calc_subresource_for_copy(&r.dst_base)
            };

            unsafe {
                list.CopyTextureRegion(
                    &dst_location,
                    r.dst_base.origin.x,
                    r.dst_base.origin.y,
                    r.dst_base.origin.z,
                    &src_location,
                    &src_box,
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
        let mut src_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            u: unsafe { mem::zeroed() },
        };
        let mut dst_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: unsafe { mem::zeroed() },
        };
        for r in regions {
            let src_box = make_box(&wgt::Origin3d::ZERO, &r.size);
            unsafe {
                *src_location.u.PlacedFootprint_mut() = r.to_subresource_footprint(dst.format)
            };
            unsafe {
                *dst_location.u.SubresourceIndex_mut() =
                    dst.calc_subresource_for_copy(&r.texture_base)
            };
            unsafe {
                list.CopyTextureRegion(
                    &dst_location,
                    r.texture_base.origin.x,
                    r.texture_base.origin.y,
                    r.texture_base.origin.z,
                    &src_location,
                    &src_box,
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
        let mut src_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: src.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            u: unsafe { mem::zeroed() },
        };
        let mut dst_location = d3d12_ty::D3D12_TEXTURE_COPY_LOCATION {
            pResource: dst.resource.as_mut_ptr(),
            Type: d3d12_ty::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            u: unsafe { mem::zeroed() },
        };
        for r in regions {
            let src_box = make_box(&r.texture_base.origin, &r.size);
            unsafe {
                *src_location.u.SubresourceIndex_mut() =
                    src.calc_subresource_for_copy(&r.texture_base)
            };
            unsafe {
                *dst_location.u.PlacedFootprint_mut() = r.to_subresource_footprint(src.format)
            };
            unsafe { list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, &src_box) };
        }
    }

    unsafe fn begin_query(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .BeginQuery(set.raw.as_mut_ptr(), set.raw_ty, index)
        };
    }
    unsafe fn end_query(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .EndQuery(set.raw.as_mut_ptr(), set.raw_ty, index)
        };
    }
    unsafe fn write_timestamp(&mut self, set: &super::QuerySet, index: u32) {
        unsafe {
            self.list.as_ref().unwrap().EndQuery(
                set.raw.as_mut_ptr(),
                d3d12_ty::D3D12_QUERY_TYPE_TIMESTAMP,
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
                set.raw.as_mut_ptr(),
                set.raw_ty,
                range.start,
                range.end - range.start,
                buffer.resource.as_mut_ptr(),
                offset,
            )
        };
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<super::Api>) {
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

        let mut color_views = [d3d12::CpuDescriptor { ptr: 0 }; crate::MAX_COLOR_ATTACHMENTS];
        for (rtv, cat) in color_views.iter_mut().zip(desc.color_attachments.iter()) {
            if let Some(cat) = cat.as_ref() {
                *rtv = cat.target.view.handle_rtv.unwrap().raw;
            } else {
                *rtv = self.null_rtv_handle.raw;
            }
        }

        let ds_view = match desc.depth_stencil_attachment {
            None => ptr::null(),
            Some(ref ds) => {
                if ds.target.usage == crate::TextureUses::DEPTH_STENCIL_WRITE {
                    &ds.target.view.handle_dsv_rw.as_ref().unwrap().raw
                } else {
                    &ds.target.view.handle_dsv_ro.as_ref().unwrap().raw
                }
            }
        };

        let list = self.list.as_ref().unwrap();
        unsafe {
            list.OMSetRenderTargets(
                desc.color_attachments.len() as u32,
                color_views.as_ptr(),
                0,
                ds_view,
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
                    list.clear_render_target_view(*rtv, value, &[]);
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
            let mut flags = d3d12::ClearFlags::empty();
            let aspects = ds.target.view.aspects;
            if !ds.depth_ops.contains(crate::AttachmentOps::LOAD)
                && aspects.contains(crate::FormatAspects::DEPTH)
            {
                flags |= d3d12::ClearFlags::DEPTH;
            }
            if !ds.stencil_ops.contains(crate::AttachmentOps::LOAD)
                && aspects.contains(crate::FormatAspects::STENCIL)
            {
                flags |= d3d12::ClearFlags::STENCIL;
            }

            if !ds_view.is_null() && !flags.is_empty() {
                list.clear_depth_stencil_view(
                    unsafe { *ds_view },
                    flags,
                    ds.clear_value.0,
                    ds.clear_value.1 as u8,
                    &[],
                );
            }
        }

        let raw_vp = d3d12_ty::D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: desc.extent.width as f32,
            Height: desc.extent.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let raw_rect = d3d12_ty::D3D12_RECT {
            left: 0,
            top: 0,
            right: desc.extent.width as i32,
            bottom: desc.extent.height as i32,
        };
        unsafe { list.RSSetViewports(1, &raw_vp) };
        unsafe { list.RSSetScissorRects(1, &raw_rect) };
    }

    unsafe fn end_render_pass(&mut self) {
        if !self.pass.resolves.is_empty() {
            let list = self.list.as_ref().unwrap();
            self.temp.barriers.clear();

            // All the targets are expected to be in `COLOR_TARGET` state,
            // but D3D12 has special source/destination states for the resolves.
            for resolve in self.pass.resolves.iter() {
                let mut barrier = d3d12_ty::D3D12_RESOURCE_BARRIER {
                    Type: d3d12_ty::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: d3d12_ty::D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    u: unsafe { mem::zeroed() },
                };
                //Note: this assumes `D3D12_RESOURCE_STATE_RENDER_TARGET`.
                // If it's not the case, we can include the `TextureUses` in `PassResove`.
                unsafe {
                    *barrier.u.Transition_mut() = d3d12_ty::D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: resolve.src.0.as_mut_ptr(),
                        Subresource: resolve.src.1,
                        StateBefore: d3d12_ty::D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: d3d12_ty::D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    }
                };
                self.temp.barriers.push(barrier);
                unsafe {
                    *barrier.u.Transition_mut() = d3d12_ty::D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: resolve.dst.0.as_mut_ptr(),
                        Subresource: resolve.dst.1,
                        StateBefore: d3d12_ty::D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: d3d12_ty::D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    }
                };
                self.temp.barriers.push(barrier);
            }

            if !self.temp.barriers.is_empty() {
                profiling::scope!("ID3D12GraphicsCommandList::ResourceBarrier");
                unsafe {
                    list.ResourceBarrier(
                        self.temp.barriers.len() as u32,
                        self.temp.barriers.as_ptr(),
                    )
                };
            }

            for resolve in self.pass.resolves.iter() {
                profiling::scope!("ID3D12GraphicsCommandList::ResolveSubresource");
                unsafe {
                    list.ResolveSubresource(
                        resolve.dst.0.as_mut_ptr(),
                        resolve.dst.1,
                        resolve.src.0.as_mut_ptr(),
                        resolve.src.1,
                        resolve.format,
                    )
                };
            }

            // Flip all the barriers to reverse, back into `COLOR_TARGET`.
            for barrier in self.temp.barriers.iter_mut() {
                let transition = unsafe { barrier.u.Transition_mut() };
                mem::swap(&mut transition.StateBefore, &mut transition.StateAfter);
            }
            if !self.temp.barriers.is_empty() {
                profiling::scope!("ID3D12GraphicsCommandList::ResourceBarrier");
                unsafe {
                    list.ResourceBarrier(
                        self.temp.barriers.len() as u32,
                        self.temp.barriers.as_ptr(),
                    )
                };
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
        log::trace!("Set group[{}]", index);
        let info = &layout.bind_group_infos[index as usize];
        let mut root_index = info.base_root_index as usize;

        // Bind CBV/SRC/UAV descriptor tables
        if info.tables.contains(super::TableTypes::SRV_CBV_UAV) {
            log::trace!("\tBind element[{}] = view", root_index);
            self.pass.root_elements[root_index] =
                super::RootElement::Table(group.handle_views.unwrap().gpu);
            root_index += 1;
        }

        // Bind Sampler descriptor tables.
        if info.tables.contains(super::TableTypes::SAMPLERS) {
            log::trace!("\tBind element[{}] = sampler", root_index);
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
            log::trace!("\tBind element[{}] = dynamic", root_index);
            self.pass.root_elements[root_index] = super::RootElement::DynamicOffsetBuffer {
                kind,
                address: gpu_base + offset as d3d12::GpuAddress,
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
                .SetMarker(0, wide_label.as_ptr() as *const _, size)
        };
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        let (wide_label, size) = self.temp.prepare_marker(group_label);
        unsafe {
            self.list
                .as_ref()
                .unwrap()
                .BeginEvent(0, wide_label.as_ptr() as *const _, size)
        };
    }
    unsafe fn end_debug_marker(&mut self) {
        unsafe { self.list.as_ref().unwrap().EndEvent() }
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &super::RenderPipeline) {
        let list = self.list.as_ref().unwrap().clone();

        if self.pass.layout.signature != pipeline.layout.signature {
            // D3D12 requires full reset on signature change
            list.set_graphics_root_signature(&pipeline.layout.signature);
            self.reset_signature(&pipeline.layout);
        };

        list.set_pipeline_state(&pipeline.raw);
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
        binding: crate::BufferBinding<'a, super::Api>,
        format: wgt::IndexFormat,
    ) {
        self.list.as_ref().unwrap().set_index_buffer(
            binding.resolve_address(),
            binding.resolve_size() as u32,
            auxil::dxgi::conv::map_index_format(format),
        );
    }
    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: crate::BufferBinding<'a, super::Api>,
    ) {
        let vb = &mut self.pass.vertex_buffers[index as usize];
        vb.BufferLocation = binding.resolve_address();
        vb.SizeInBytes = binding.resolve_size() as u32;
        self.pass.dirty_vertex_buffers |= 1 << index;
    }

    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {
        let raw_vp = d3d12_ty::D3D12_VIEWPORT {
            TopLeftX: rect.x,
            TopLeftY: rect.y,
            Width: rect.w,
            Height: rect.h,
            MinDepth: depth_range.start,
            MaxDepth: depth_range.end,
        };
        unsafe { self.list.as_ref().unwrap().RSSetViewports(1, &raw_vp) };
    }
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        let raw_rect = d3d12_ty::D3D12_RECT {
            left: rect.x as i32,
            top: rect.y as i32,
            right: (rect.x + rect.w) as i32,
            bottom: (rect.y + rect.h) as i32,
        };
        unsafe { self.list.as_ref().unwrap().RSSetScissorRects(1, &raw_rect) };
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.list.as_ref().unwrap().set_stencil_reference(value);
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        self.list.as_ref().unwrap().set_blend_factor(*color);
    }

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        unsafe { self.prepare_draw(start_vertex as i32, start_instance) };
        self.list.as_ref().unwrap().draw(
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
        unsafe { self.prepare_draw(base_vertex, start_instance) };
        self.list.as_ref().unwrap().draw_indexed(
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
        unsafe { self.prepare_draw(0, 0) };
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                self.shared.cmd_signatures.draw.as_mut_ptr(),
                draw_count,
                buffer.resource.as_mut_ptr(),
                offset,
                ptr::null_mut(),
                0,
            )
        };
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
                self.shared.cmd_signatures.draw_indexed.as_mut_ptr(),
                draw_count,
                buffer.resource.as_mut_ptr(),
                offset,
                ptr::null_mut(),
                0,
            )
        };
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
                self.shared.cmd_signatures.draw.as_mut_ptr(),
                max_count,
                buffer.resource.as_mut_ptr(),
                offset,
                count_buffer.resource.as_mut_ptr(),
                count_offset,
            )
        };
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
                self.shared.cmd_signatures.draw_indexed.as_mut_ptr(),
                max_count,
                buffer.resource.as_mut_ptr(),
                offset,
                count_buffer.resource.as_mut_ptr(),
                count_offset,
            )
        };
    }

    // compute

    unsafe fn begin_compute_pass<'a>(
        &mut self,
        desc: &crate::ComputePassDescriptor<'a, super::Api>,
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
        let list = self.list.as_ref().unwrap().clone();

        if self.pass.layout.signature != pipeline.layout.signature {
            // D3D12 requires full reset on signature change
            list.set_compute_root_signature(&pipeline.layout.signature);
            self.reset_signature(&pipeline.layout);
        };

        list.set_pipeline_state(&pipeline.raw);
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        self.prepare_dispatch(count);
        self.list.as_ref().unwrap().dispatch(count);
    }
    unsafe fn dispatch_indirect(&mut self, buffer: &super::Buffer, offset: wgt::BufferAddress) {
        self.prepare_dispatch([0; 3]);
        //TODO: update special constants indirectly
        unsafe {
            self.list.as_ref().unwrap().ExecuteIndirect(
                self.shared.cmd_signatures.dispatch.as_mut_ptr(),
                1,
                buffer.resource.as_mut_ptr(),
                offset,
                ptr::null_mut(),
                0,
            )
        };
    }
}
