/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::identity::IdentityRecyclerFactory;

use wgc::{gfx_select, id};

use std::slice;

pub type Global = wgc::hub::Global<IdentityRecyclerFactory>;

#[no_mangle]
pub extern "C" fn wgpu_server_new(factory: IdentityRecyclerFactory) -> *mut Global {
    log::info!("Initializing WGPU server");
    Box::into_raw(Box::new(Global::new("wgpu", factory)))
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_delete(global: *mut Global) {
    log::info!("Terminating WGPU server");
    Box::from_raw(global).delete();
    log::info!("\t...done");
}

#[no_mangle]
pub extern "C" fn wgpu_server_poll_all_devices(global: &Global, force_wait: bool) {
    global.poll_all_devices(force_wait);
}

/// Request an adapter according to the specified options.
/// Provide the list of IDs to pick from.
///
/// Returns the index in this list, or -1 if unable to pick.
///
/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `id_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_instance_request_adapter(
    global: &Global,
    desc: &wgc::instance::RequestAdapterOptions,
    ids: *const id::AdapterId,
    id_length: usize,
) -> i8 {
    let ids = slice::from_raw_parts(ids, id_length);
    match global.pick_adapter(
        desc,
        wgc::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Some(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_request_device(
    global: &Global,
    self_id: id::AdapterId,
    desc: &wgt::DeviceDescriptor,
    new_id: id::DeviceId,
) {
    gfx_select!(self_id => global.adapter_request_device(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_destroy(global: &Global, adapter_id: id::AdapterId) {
    gfx_select!(adapter_id => global.adapter_destroy(adapter_id))
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_destroy(global: &Global, self_id: id::DeviceId) {
    gfx_select!(self_id => global.device_destroy(self_id))
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_buffer(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::BufferDescriptor,
    new_id: id::BufferId,
) {
    gfx_select!(self_id => global.device_create_buffer(self_id, desc, new_id));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `size` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_device_set_buffer_sub_data(
    global: &Global,
    self_id: id::DeviceId,
    buffer_id: id::BufferId,
    offset: wgt::BufferAddress,
    data: *const u8,
    size: wgt::BufferAddress,
) {
    let slice = slice::from_raw_parts(data, size as usize);
    gfx_select!(self_id => global.device_set_buffer_sub_data(self_id, buffer_id, offset, slice));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `size` elements.
#[no_mangle]
pub extern "C" fn wgpu_server_buffer_map_read(
    global: &Global,
    buffer_id: id::BufferId,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
    callback: wgc::device::BufferMapReadCallback,
    userdata: *mut u8,
) {
    let operation = wgc::resource::BufferMapOperation::Read { callback, userdata };

    gfx_select!(buffer_id => global.buffer_map_async(
        buffer_id,
        start .. start + size,
        operation
    ));
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_unmap(global: &Global, buffer_id: id::BufferId) {
    gfx_select!(buffer_id => global.buffer_unmap(buffer_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_destroy(global: &Global, self_id: id::BufferId) {
    gfx_select!(self_id => global.buffer_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_encoder(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::CommandEncoderDescriptor,
    new_id: id::CommandEncoderId,
) {
    gfx_select!(self_id => global.device_create_command_encoder(self_id, &desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_finish(
    global: &Global,
    self_id: id::CommandEncoderId,
    desc: &wgt::CommandBufferDescriptor,
) {
    gfx_select!(self_id => global.command_encoder_finish(self_id, desc));
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_destroy(global: &Global, self_id: id::CommandEncoderId) {
    gfx_select!(self_id => global.command_encoder_destroy(self_id));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `byte_length` elements.
#[no_mangle]
pub extern "C" fn wgpu_server_command_buffer_destroy(
    global: &Global,
    self_id: id::CommandBufferId,
) {
    gfx_select!(self_id => global.command_buffer_destroy(self_id));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_buffer_to_buffer(
    global: &Global,
    self_id: id::CommandEncoderId,
    source_id: id::BufferId,
    source_offset: wgt::BufferAddress,
    destination_id: id::BufferId,
    destination_offset: wgt::BufferAddress,
    size: wgt::BufferAddress,
) {
    gfx_select!(self_id => global.command_encoder_copy_buffer_to_buffer(self_id, source_id, source_offset, destination_id, destination_offset, size));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_texture_to_buffer(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::TextureCopyView,
    destination: &wgc::command::BufferCopyView,
    size: wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_texture_to_buffer(self_id, source, destination, size));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_buffer_to_texture(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::BufferCopyView,
    destination: &wgc::command::TextureCopyView,
    size: wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_buffer_to_texture(self_id, source, destination, size));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_texture_to_texture(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::TextureCopyView,
    destination: &wgc::command::TextureCopyView,
    size: wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_texture_to_texture(self_id, source, destination, size));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointers are
/// valid for `color_attachments_length` and `command_length` elements,
/// respectively.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encode_compute_pass(
    global: &Global,
    self_id: id::CommandEncoderId,
    bytes: *const u8,
    byte_length: usize,
) {
    let raw_data = slice::from_raw_parts(bytes, byte_length);
    gfx_select!(self_id => global.command_encoder_run_compute_pass(self_id, raw_data));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointers are
/// valid for `color_attachments_length` and `command_length` elements,
/// respectively.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encode_render_pass(
    global: &Global,
    self_id: id::CommandEncoderId,
    commands: *const u8,
    command_length: usize,
) {
    let raw_pass = slice::from_raw_parts(commands, command_length);
    gfx_select!(self_id => global.command_encoder_run_render_pass(self_id, raw_pass));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `command_buffer_id_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_submit(
    global: &Global,
    self_id: id::QueueId,
    command_buffer_ids: *const id::CommandBufferId,
    command_buffer_id_length: usize,
) {
    let command_buffers = slice::from_raw_parts(command_buffer_ids, command_buffer_id_length);
    gfx_select!(self_id => global.queue_submit(self_id, command_buffers));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_bind_group_layout(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::binding_model::BindGroupLayoutDescriptor,
    new_id: id::BindGroupLayoutId,
) {
    gfx_select!(self_id => global.device_create_bind_group_layout(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_layout_destroy(
    global: &Global,
    self_id: id::BindGroupLayoutId,
) {
    gfx_select!(self_id => global.bind_group_layout_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_pipeline_layout(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::binding_model::PipelineLayoutDescriptor,
    new_id: id::PipelineLayoutId,
) {
    gfx_select!(self_id => global.device_create_pipeline_layout(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_pipeline_layout_destroy(
    global: &Global,
    self_id: id::PipelineLayoutId,
) {
    gfx_select!(self_id => global.pipeline_layout_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_bind_group(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::binding_model::BindGroupDescriptor,
    new_id: id::BindGroupId,
) {
    gfx_select!(self_id => global.device_create_bind_group(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_destroy(global: &Global, self_id: id::BindGroupId) {
    gfx_select!(self_id => global.bind_group_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_shader_module(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::pipeline::ShaderModuleDescriptor,
    new_id: id::ShaderModuleId,
) {
    gfx_select!(self_id => global.device_create_shader_module(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_shader_module_destroy(global: &Global, self_id: id::ShaderModuleId) {
    gfx_select!(self_id => global.shader_module_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_compute_pipeline(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::pipeline::ComputePipelineDescriptor,
    new_id: id::ComputePipelineId,
) {
    gfx_select!(self_id => global.device_create_compute_pipeline(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_compute_pipeline_destroy(
    global: &Global,
    self_id: id::ComputePipelineId,
) {
    gfx_select!(self_id => global.compute_pipeline_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_render_pipeline(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgc::pipeline::RenderPipelineDescriptor,
    new_id: id::RenderPipelineId,
) {
    gfx_select!(self_id => global.device_create_render_pipeline(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_render_pipeline_destroy(
    global: &Global,
    self_id: id::RenderPipelineId,
) {
    gfx_select!(self_id => global.render_pipeline_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_texture(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::TextureDescriptor,
    new_id: id::TextureId,
) {
    gfx_select!(self_id => global.device_create_texture(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_create_view(
    global: &Global,
    self_id: id::TextureId,
    desc: Option<&wgt::TextureViewDescriptor>,
    new_id: id::TextureViewId,
) {
    gfx_select!(self_id => global.texture_create_view(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_destroy(global: &Global, self_id: id::TextureId) {
    gfx_select!(self_id => global.texture_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_view_destroy(global: &Global, self_id: id::TextureViewId) {
    gfx_select!(self_id => global.texture_view_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_sampler(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::SamplerDescriptor,
    new_id: id::SamplerId,
) {
    gfx_select!(self_id => global.device_create_sampler(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_sampler_destroy(global: &Global, self_id: id::SamplerId) {
    gfx_select!(self_id => global.sampler_destroy(self_id));
}
