/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use wgc::{hub::IdentityManager, id};

use wgt::Backend;

pub use wgc::command::{compute_ffi::*, render_ffi::*};

use parking_lot::Mutex;

use std::{ptr, slice};

pub mod identity;
pub mod server;

#[derive(Debug, Default)]
struct IdentityHub {
    adapters: IdentityManager,
    devices: IdentityManager,
    buffers: IdentityManager,
    command_buffers: IdentityManager,
    bind_group_layouts: IdentityManager,
    pipeline_layouts: IdentityManager,
    bind_groups: IdentityManager,
    shader_modules: IdentityManager,
    compute_pipelines: IdentityManager,
    render_pipelines: IdentityManager,
    textures: IdentityManager,
    texture_views: IdentityManager,
    samplers: IdentityManager,
}

#[derive(Debug, Default)]
struct Identities {
    surfaces: IdentityManager,
    vulkan: IdentityHub,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    metal: IdentityHub,
    #[cfg(windows)]
    dx12: IdentityHub,
}

impl Identities {
    fn select(&mut self, backend: Backend) -> &mut IdentityHub {
        match backend {
            Backend::Vulkan => &mut self.vulkan,
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            Backend::Metal => &mut self.metal,
            #[cfg(windows)]
            Backend::Dx12 => &mut self.dx12,
            _ => panic!("Unexpected backend: {:?}", backend),
        }
    }
}

#[derive(Debug)]
pub struct Client {
    identities: Mutex<Identities>,
}

#[repr(C)]
#[derive(Debug)]
pub struct Infrastructure {
    pub client: *mut Client,
    pub error: *const u8,
}

#[no_mangle]
pub extern "C" fn wgpu_client_new() -> Infrastructure {
    log::info!("Initializing WGPU client");
    let client = Box::new(Client {
        identities: Mutex::new(Identities::default()),
    });
    Infrastructure {
        client: Box::into_raw(client),
        error: ptr::null(),
    }
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_client_delete(client: *mut Client) {
    log::info!("Terminating WGPU client");
    let _client = Box::from_raw(client);
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `id_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_client_make_adapter_ids(
    client: &Client,
    ids: *mut id::AdapterId,
    id_length: usize,
) -> usize {
    let mut identities = client.identities.lock();
    assert_ne!(id_length, 0);
    let mut ids = slice::from_raw_parts_mut(ids, id_length).iter_mut();

    *ids.next().unwrap() = identities.vulkan.adapters.alloc(Backend::Vulkan);

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        *ids.next().unwrap() = identities.metal.adapters.alloc(Backend::Metal);
    }
    #[cfg(windows)]
    {
        *ids.next().unwrap() = identities.dx12.adapters.alloc(Backend::Dx12);
    }

    id_length - ids.len()
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_adapter_id(client: &Client, id: id::AdapterId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .adapters
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_device_id(
    client: &Client,
    adapter_id: id::AdapterId,
) -> id::DeviceId {
    let backend = adapter_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .devices
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_device_id(client: &Client, id: id::DeviceId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .devices
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_buffer_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::BufferId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .buffers
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_buffer_id(client: &Client, id: id::BufferId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .buffers
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_texture_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::TextureId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .textures
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_texture_id(client: &Client, id: id::TextureId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .textures
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_texture_view_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::TextureViewId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .texture_views
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_texture_view_id(client: &Client, id: id::TextureViewId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .texture_views
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_sampler_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::SamplerId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .samplers
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_sampler_id(client: &Client, id: id::SamplerId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .samplers
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_encoder_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::CommandEncoderId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .command_buffers
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_encoder_id(client: &Client, id: id::CommandEncoderId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .command_buffers
        .free(id)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_compute_pass(
    encoder_id: id::CommandEncoderId,
    _desc: Option<&wgc::command::ComputePassDescriptor>,
) -> wgc::command::RawPass {
    wgc::command::RawPass::new_compute(encoder_id)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_compute_pass_destroy(pass: wgc::command::RawPass) {
    let _ = pass.into_vec();
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_render_pass(
    encoder_id: id::CommandEncoderId,
    desc: &wgc::command::RenderPassDescriptor,
) -> wgc::command::RawPass {
    wgc::command::RawPass::new_render(encoder_id, desc)
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_render_pass_destroy(pass: wgc::command::RawPass) {
    let _ = pass.into_vec();
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_bind_group_layout_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::BindGroupLayoutId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .bind_group_layouts
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_bind_group_layout_id(
    client: &Client,
    id: id::BindGroupLayoutId,
) {
    client
        .identities
        .lock()
        .select(id.backend())
        .bind_group_layouts
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_pipeline_layout_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::PipelineLayoutId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .pipeline_layouts
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_pipeline_layout_id(client: &Client, id: id::PipelineLayoutId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .pipeline_layouts
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_bind_group_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::BindGroupId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .bind_groups
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_bind_group_id(client: &Client, id: id::BindGroupId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .bind_groups
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_shader_module_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::ShaderModuleId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .shader_modules
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_shader_module_id(client: &Client, id: id::ShaderModuleId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .shader_modules
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_compute_pipeline_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::ComputePipelineId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .compute_pipelines
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_compute_pipeline_id(client: &Client, id: id::ComputePipelineId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .compute_pipelines
        .free(id)
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_render_pipeline_id(
    client: &Client,
    device_id: id::DeviceId,
) -> id::RenderPipelineId {
    let backend = device_id.backend();
    client
        .identities
        .lock()
        .select(backend)
        .render_pipelines
        .alloc(backend)
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_render_pipeline_id(client: &Client, id: id::RenderPipelineId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .render_pipelines
        .free(id)
}
