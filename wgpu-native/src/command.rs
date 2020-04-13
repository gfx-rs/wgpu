/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::GLOBAL;

pub use core::command::{compute_ffi::*, render_ffi::*};

use core::{gfx_select, id};

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_finish(
    encoder_id: id::CommandEncoderId,
    desc: Option<&wgt::CommandBufferDescriptor>,
) -> id::CommandBufferId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(encoder_id => GLOBAL.command_encoder_finish(encoder_id, desc))
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_buffer_to_buffer(
    command_encoder_id: id::CommandEncoderId,
    source: id::BufferId,
    source_offset: wgt::BufferAddress,
    destination: id::BufferId,
    destination_offset: wgt::BufferAddress,
    size: wgt::BufferAddress,
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
    copy_size: wgt::Extent3d,
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
    copy_size: wgt::Extent3d,
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
    copy_size: wgt::Extent3d,
) {
    gfx_select!(command_encoder_id => GLOBAL.command_encoder_copy_texture_to_texture(
        command_encoder_id,
        source,
        destination,
        copy_size))
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_render_pass(
    encoder_id: id::CommandEncoderId,
    desc: &core::command::RenderPassDescriptor,
) -> *mut core::command::RawPass {
    let pass = core::command::RawPass::new_render(encoder_id, desc);
    Box::into_raw(Box::new(pass))
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_render_pass_end_pass(pass_id: id::RenderPassId) {
    let (pass_data, encoder_id) = Box::from_raw(pass_id).finish_render();
    gfx_select!(encoder_id => GLOBAL.command_encoder_run_render_pass(encoder_id, &pass_data))
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_render_pass_destroy(pass: *mut core::command::RawPass) {
    let _ = Box::from_raw(pass).into_vec();
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_compute_pass(
    encoder_id: id::CommandEncoderId,
    _desc: Option<&core::command::ComputePassDescriptor>,
) -> *mut core::command::RawPass {
    let pass = core::command::RawPass::new_compute(encoder_id);
    Box::into_raw(Box::new(pass))
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_compute_pass_end_pass(pass_id: id::ComputePassId) {
    let (pass_data, encoder_id) = Box::from_raw(pass_id).finish_compute();
    gfx_select!(encoder_id => GLOBAL.command_encoder_run_compute_pass(encoder_id, &pass_data))
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_compute_pass_destroy(pass: *mut core::command::RawPass) {
    let _ = Box::from_raw(pass).into_vec();
}
