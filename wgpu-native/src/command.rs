/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::GLOBAL;

use core::{gfx_select, id};

use std::{marker::PhantomData, slice};


#[no_mangle]
pub extern "C" fn wgpu_command_encoder_finish(
    encoder_id: id::CommandEncoderId,
    desc: Option<&core::command::CommandBufferDescriptor>,
) -> id::CommandBufferId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(encoder_id => GLOBAL.command_encoder_finish(encoder_id, desc))
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_buffer_to_buffer(
    command_encoder_id: id::CommandEncoderId,
    source: id::BufferId,
    source_offset: core::BufferAddress,
    destination: id::BufferId,
    destination_offset: core::BufferAddress,
    size: core::BufferAddress,
) {
    gfx_select!(command_encoder_id => GLOBAL.command_encoder_copy_buffer_to_buffer(
        command_encoder_id,
        source, source_offset,
        destination,
        destination_offset,
        size))
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_buffer_to_texture(
    command_encoder_id: id::CommandEncoderId,
    source: &core::command::BufferCopyView,
    destination: &core::command::TextureCopyView,
    copy_size: core::Extent3d,
) {
    gfx_select!(command_encoder_id => GLOBAL.command_encoder_copy_buffer_to_texture(
        command_encoder_id,
        source,
        destination,
        copy_size))
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_texture_to_buffer(
    command_encoder_id: id::CommandEncoderId,
    source: &core::command::TextureCopyView,
    destination: &core::command::BufferCopyView,
    copy_size: core::Extent3d,
) {
    gfx_select!(command_encoder_id => GLOBAL.command_encoder_copy_texture_to_buffer(
        command_encoder_id,
        source,
        destination,
        copy_size))
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_texture_to_texture(
    command_encoder_id: id::CommandEncoderId,
    source: &core::command::TextureCopyView,
    destination: &core::command::TextureCopyView,
    copy_size: core::Extent3d,
) {
    gfx_select!(command_encoder_id => GLOBAL.command_encoder_copy_texture_to_texture(
        command_encoder_id,
        source,
        destination,
        copy_size))
}


#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_render_pass(
    encoder_id: id::CommandEncoderId,
    desc: &core::command::RenderPassDescriptor,
) -> id::RenderPassId {
    gfx_select!(encoder_id => GLOBAL.command_encoder_begin_render_pass(encoder_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(pass_id: id::RenderPassId) {
    gfx_select!(pass_id => GLOBAL.render_pass_end_pass(pass_id))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_bind_group(
    pass_id: id::RenderPassId,
    index: u32,
    bind_group_id: id::BindGroupId,
    offsets: *const core::BufferAddress,
    offsets_length: usize,
) {
    let offsets = if offsets_length != 0 {
        unsafe { slice::from_raw_parts(offsets, offsets_length) }
    } else {
        &[]
    };
    gfx_select!(pass_id => GLOBAL.render_pass_set_bind_group(pass_id, index, bind_group_id, offsets))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_push_debug_group(
    _pass_id: id::RenderPassId,
    _label: core::RawString,
) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_pop_debug_group(_pass_id: id::RenderPassId) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_insert_debug_marker(
    _pass_id: id::RenderPassId,
    _label: core::RawString,
) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_index_buffer(
    pass_id: id::RenderPassId,
    buffer_id: id::BufferId,
    offset: core::BufferAddress,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_index_buffer(pass_id, buffer_id, offset))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_vertex_buffers(
    pass_id: id::RenderPassId,
    start_slot: u32,
    buffers: *const id::BufferId,
    offsets: *const core::BufferAddress,
    length: usize,
) {
    let buffers = unsafe { slice::from_raw_parts(buffers, length) };
    let offsets = unsafe { slice::from_raw_parts(offsets, length) };
    gfx_select!(pass_id => GLOBAL.render_pass_set_vertex_buffers(pass_id, start_slot, buffers, offsets))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw(
    pass_id: id::RenderPassId,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_draw(pass_id, vertex_count, instance_count, first_vertex, first_instance))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw_indirect(
    pass_id: id::RenderPassId,
    indirect_buffer_id: id::BufferId,
    indirect_offset: core::BufferAddress,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_draw_indirect(pass_id, indirect_buffer_id, indirect_offset))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw_indexed(
    pass_id: id::RenderPassId,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_draw_indexed(pass_id, index_count, instance_count, first_index, base_vertex, first_instance))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw_indexed_indirect(
    pass_id: id::RenderPassId,
    indirect_buffer_id: id::BufferId,
    indirect_offset: core::BufferAddress,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_draw_indexed_indirect(pass_id, indirect_buffer_id, indirect_offset))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_pipeline(
    pass_id: id::RenderPassId,
    pipeline_id: id::RenderPipelineId,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_pipeline(pass_id, pipeline_id))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_blend_color(pass_id: id::RenderPassId, color: &core::Color) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_blend_color(pass_id, color))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_stencil_reference(pass_id: id::RenderPassId, value: u32) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_stencil_reference(pass_id, value))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_viewport(
    pass_id: id::RenderPassId,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    min_depth: f32,
    max_depth: f32,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_viewport(pass_id, x, y, w, h, min_depth, max_depth))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_scissor_rect(
    pass_id: id::RenderPassId,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    gfx_select!(pass_id => GLOBAL.render_pass_set_scissor_rect(pass_id, x, y, w, h))
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_execute_bundles(
    _pass_id: id::RenderPassId,
    _bundles: *const id::RenderBundleId,
    _bundles_length: usize,
) {
    unimplemented!()
}


#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_compute_pass(
    encoder_id: id::CommandEncoderId,
    desc: Option<&core::command::ComputePassDescriptor>,
) -> id::ComputePassId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(encoder_id => GLOBAL.command_encoder_begin_compute_pass(encoder_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_end_pass(pass_id: id::ComputePassId) {
    gfx_select!(pass_id => GLOBAL.compute_pass_end_pass(pass_id))
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: id::ComputePassId,
    index: u32,
    bind_group_id: id::BindGroupId,
    offsets: *const core::BufferAddress,
    offsets_length: usize,
) {
    let offsets = if offsets_length != 0 {
        unsafe { slice::from_raw_parts(offsets, offsets_length) }
    } else {
        &[]
    };
    gfx_select!(pass_id => GLOBAL.compute_pass_set_bind_group(pass_id, index, bind_group_id, offsets))
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_push_debug_group(
    _pass_id: id::ComputePassId,
    _label: core::RawString,
) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_pop_debug_group(_pass_id: id::ComputePassId) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_insert_debug_marker(
    _pass_id: id::ComputePassId,
    _label: core::RawString,
) {
    //TODO
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_dispatch(pass_id: id::ComputePassId, x: u32, y: u32, z: u32) {
    gfx_select!(pass_id => GLOBAL.compute_pass_dispatch(pass_id, x, y, z))
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_dispatch_indirect(
    pass_id: id::ComputePassId,
    indirect_buffer_id: id::BufferId,
    indirect_offset: core::BufferAddress,
) {
    gfx_select!(pass_id => GLOBAL.compute_pass_dispatch_indirect(pass_id, indirect_buffer_id, indirect_offset))
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: id::ComputePassId,
    pipeline_id: id::ComputePipelineId,
) {
    gfx_select!(pass_id => GLOBAL.compute_pass_set_pipeline(pass_id, pipeline_id))
}
