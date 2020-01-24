/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use core::{gfx_select, id};

use std::slice;

pub type Global = core::hub::Global<()>;

#[no_mangle]
pub extern "C" fn wgpu_server_new() -> *mut Global {
    log::info!("Initializing WGPU server");
    Box::into_raw(Box::new(Global::new("wgpu")))
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
    desc: &core::instance::RequestAdapterOptions,
    ids: *const id::AdapterId,
    id_length: usize,
) -> i8 {
    let ids = slice::from_raw_parts(ids, id_length);
    match global.pick_adapter(
        desc,
        core::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Some(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_request_device(
    global: &Global,
    self_id: id::AdapterId,
    desc: &core::instance::DeviceDescriptor,
    new_id: id::DeviceId,
) {
    gfx_select!(self_id => global.adapter_request_device(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_destroy(
    global: &Global,
    adapter_id: id::AdapterId,
) {
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
    desc: &core::resource::BufferDescriptor,
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
    offset: core::BufferAddress,
    data: *const u8,
    size: core::BufferAddress,
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
    start: core::BufferAddress,
    size: core::BufferAddress,
    callback: core::device::BufferMapReadCallback,
    userdata: *mut u8,
) {
    let operation = core::resource::BufferMapOperation::Read(
        Box::new(move |status, data| unsafe {
            callback(status, data, userdata)
        }),
    );
    gfx_select!(buffer_id => global.buffer_map_async(
        buffer_id,
        core::resource::BufferUsage::MAP_READ,
        start .. start + size,
        operation
    ));
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_destroy(global: &Global, self_id: id::BufferId) {
    gfx_select!(self_id => global.buffer_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_encoder(
    global: &Global,
    self_id: id::DeviceId,
    desc: &core::command::CommandEncoderDescriptor,
    new_id: id::CommandEncoderId,
) {
    gfx_select!(self_id => global.device_create_command_encoder(self_id, &desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_finish(
    global: &Global,
    self_id: id::CommandEncoderId,
    desc: &core::command::CommandBufferDescriptor,
) {
    gfx_select!(self_id => global.command_encoder_finish(self_id, desc));
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_destroy(
    global: &Global,
    self_id: id::CommandEncoderId,
) {
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
    source_offset: core::BufferAddress,
    destination_id: id::BufferId,
    destination_offset: core::BufferAddress,
    size: core::BufferAddress,
) {
    gfx_select!(self_id => global.command_encoder_copy_buffer_to_buffer(self_id, source_id, source_offset, destination_id, destination_offset, size));
}

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
    color_attachments: *const core::command::RenderPassColorAttachmentDescriptor,
    color_attachment_length: usize,
    depth_stencil_attachment: Option<&core::command::RenderPassDepthStencilAttachmentDescriptor>,
    commands: *const u8,
    command_length: usize,
) {
    let color_attachments = slice::from_raw_parts(color_attachments, color_attachment_length);
    let raw_pass = slice::from_raw_parts(commands, command_length);
    gfx_select!(self_id => global.command_encoder_run_render_pass(self_id, color_attachments, depth_stencil_attachment, raw_pass));
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
    desc: &core::binding_model::BindGroupLayoutDescriptor,
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
    desc: &core::binding_model::PipelineLayoutDescriptor,
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
    desc: &core::binding_model::BindGroupDescriptor,
    new_id: id::BindGroupId,
) {
    gfx_select!(self_id => global.device_create_bind_group(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_destroy(
    global: &Global,
    self_id: id::BindGroupId,
) {
    gfx_select!(self_id => global.bind_group_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_shader_module(
    global: &Global,
    self_id: id::DeviceId,
    desc: &core::pipeline::ShaderModuleDescriptor,
    new_id: id::ShaderModuleId,
) {
    gfx_select!(self_id => global.device_create_shader_module(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_shader_module_destroy(
    global: &Global,
    self_id: id::ShaderModuleId,
) {
    gfx_select!(self_id => global.shader_module_destroy(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_compute_pipeline(
    global: &Global,
    self_id: id::DeviceId,
    desc: &core::pipeline::ComputePipelineDescriptor,
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
