/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::GLOBAL;

use core::{gfx_select, hub::Token, id};

use std::{marker::PhantomData, slice};

pub type RequestAdapterCallback =
    unsafe extern "C" fn(id: id::AdapterId, userdata: *mut std::ffi::c_void);
pub type BufferMapReadCallback =
    unsafe extern "C" fn(status: core::resource::BufferMapAsyncStatus, data: *const u8, userdata: *mut u8);
pub type BufferMapWriteCallback =
    unsafe extern "C" fn(status: core::resource::BufferMapAsyncStatus, data: *mut u8, userdata: *mut u8);

pub fn wgpu_create_surface(raw_handle: raw_window_handle::RawWindowHandle) -> id::SurfaceId {
    use raw_window_handle::RawWindowHandle as Rwh;

    let instance = &GLOBAL.instance;
    let surface = match raw_handle {
        #[cfg(target_os = "ios")]
        Rwh::IOS(h) => core::instance::Surface {
            #[cfg(feature = "vulkan-portability")]
            vulkan: None,
            metal: instance
                .metal
                .create_surface_from_uiview(h.ui_view, cfg!(debug_assertions)),
        },
        #[cfg(target_os = "macos")]
        Rwh::MacOS(h) => core::instance::Surface {
            #[cfg(feature = "vulkan-portability")]
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_ns_view(h.ns_view)),
            metal: instance
                .metal
                .create_surface_from_nsview(h.ns_view, cfg!(debug_assertions)),
        },
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "macos")))]
        Rwh::Xlib(h) => core::instance::Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_xlib(h.display as _, h.window as _)),
        },
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "macos")))]
        Rwh::Wayland(h) => core::instance::Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_wayland(h.display, h.surface)),
        },
        #[cfg(windows)]
        Rwh::Windows(h) => core::instance::Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_hwnd(std::ptr::null_mut(), h.hwnd)),
            dx12: instance
                .dx12
                .as_ref()
                .map(|inst| inst.create_surface_from_hwnd(h.hwnd)),
            dx11: instance.dx11.create_surface_from_hwnd(h.hwnd),
        },
        _ => panic!("Unsupported window handle"),
    };

    let mut token = Token::root();
    GLOBAL
        .surfaces
        .register_identity(PhantomData, surface, &mut token)
}

#[cfg(all(unix, not(target_os = "ios"), not(target_os = "macos")))]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_xlib(
    display: *mut *const std::ffi::c_void,
    window: u64,
) -> id::SurfaceId {
    use raw_window_handle::unix::XlibHandle;
    wgpu_create_surface(raw_window_handle::RawWindowHandle::Xlib(XlibHandle {
        window,
        display: display as *mut _,
        ..XlibHandle::empty()
    }))
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_metal_layer(
    layer: *mut std::ffi::c_void,
) -> id::SurfaceId {
    let surface = core::instance::Surface {
        #[cfg(feature = "vulkan-portability")]
        vulkan: None, //TODO: currently requires `NSView`
        metal: GLOBAL
            .instance
            .metal
            .create_surface_from_layer(layer as *mut _, cfg!(debug_assertions)),
    };

    GLOBAL
        .surfaces
        .register_identity(PhantomData, surface, &mut Token::root())
}

#[cfg(windows)]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_windows_hwnd(
    _hinstance: *mut std::ffi::c_void,
    hwnd: *mut std::ffi::c_void,
) -> id::SurfaceId {
    use raw_window_handle::windows::WindowsHandle;
    wgpu_create_surface(raw_window_handle::RawWindowHandle::Windows(
        raw_window_handle::windows::WindowsHandle {
            hwnd,
            ..WindowsHandle::empty()
        },
    ))
}

#[no_mangle]
pub extern "C" fn wgpu_request_adapter_async(
    desc: Option<&core::instance::RequestAdapterOptions>,
    mask: core::instance::BackendBit,
    callback: RequestAdapterCallback,
    userdata: *mut std::ffi::c_void,
) {
    let id = GLOBAL.pick_adapter(
        &desc.cloned().unwrap_or_default(),
        core::instance::AdapterInputs::Mask(mask, || PhantomData),
    );
    unsafe {
        callback(
            id.unwrap_or(id::AdapterId::ERROR),
            userdata,
        )
    };
}

#[no_mangle]
pub extern "C" fn wgpu_adapter_request_device(
    adapter_id: id::AdapterId,
    desc: Option<&core::instance::DeviceDescriptor>,
) -> id::DeviceId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(adapter_id => GLOBAL.adapter_request_device(adapter_id, desc, PhantomData))
}

pub fn wgpu_adapter_get_info(adapter_id: id::AdapterId) -> core::instance::AdapterInfo {
    gfx_select!(adapter_id => GLOBAL.adapter_get_info(adapter_id))
}

#[no_mangle]
pub extern "C" fn wgpu_device_get_limits(
    _device_id: id::DeviceId,
    limits: &mut core::instance::Limits,
) {
    *limits = core::instance::Limits::default(); // TODO
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer(
    device_id: id::DeviceId,
    desc: &core::resource::BufferDescriptor,
) -> id::BufferId {
    gfx_select!(device_id => GLOBAL.device_create_buffer(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer_mapped(
    device_id: id::DeviceId,
    desc: &core::resource::BufferDescriptor,
    mapped_ptr_out: *mut *mut u8,
) -> id::BufferId {
    gfx_select!(device_id => GLOBAL.device_create_buffer_mapped(device_id, desc, mapped_ptr_out, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_destroy(buffer_id: id::BufferId) {
    gfx_select!(buffer_id => GLOBAL.buffer_destroy(buffer_id))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_texture(
    device_id: id::DeviceId,
    desc: &core::resource::TextureDescriptor,
) -> id::TextureId {
    gfx_select!(device_id => GLOBAL.device_create_texture(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_texture_destroy(texture_id: id::TextureId) {
    gfx_select!(texture_id => GLOBAL.texture_destroy(texture_id))
}

#[no_mangle]
pub extern "C" fn wgpu_texture_create_view(
    texture_id: id::TextureId,
    desc: Option<&core::resource::TextureViewDescriptor>,
) -> id::TextureViewId {
    gfx_select!(texture_id => GLOBAL.texture_create_view(texture_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_texture_view_destroy(texture_view_id: id::TextureViewId) {
    gfx_select!(texture_view_id => GLOBAL.texture_view_destroy(texture_view_id))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_sampler(
    device_id: id::DeviceId,
    desc: &core::resource::SamplerDescriptor,
) -> id::SamplerId {
    gfx_select!(device_id => GLOBAL.device_create_sampler(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_sampler_destroy(sampler_id: id::SamplerId) {
    gfx_select!(sampler_id => GLOBAL.sampler_destroy(sampler_id))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group_layout(
    device_id: id::DeviceId,
    desc: &core::binding_model::BindGroupLayoutDescriptor,
) -> id::BindGroupLayoutId {
    gfx_select!(device_id => GLOBAL.device_create_bind_group_layout(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_pipeline_layout(
    device_id: id::DeviceId,
    desc: &core::binding_model::PipelineLayoutDescriptor,
) -> id::PipelineLayoutId {
    gfx_select!(device_id => GLOBAL.device_create_pipeline_layout(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group(
    device_id: id::DeviceId,
    desc: &core::binding_model::BindGroupDescriptor,
) -> id::BindGroupId {
    gfx_select!(device_id => GLOBAL.device_create_bind_group(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_bind_group_destroy(bind_group_id: id::BindGroupId) {
    gfx_select!(bind_group_id => GLOBAL.bind_group_destroy(bind_group_id))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_shader_module(
    device_id: id::DeviceId,
    desc: &core::pipeline::ShaderModuleDescriptor,
) -> id::ShaderModuleId {
    gfx_select!(device_id => GLOBAL.device_create_shader_module(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_command_encoder(
    device_id: id::DeviceId,
    desc: Option<&core::command::CommandEncoderDescriptor>,
) -> id::CommandEncoderId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(device_id => GLOBAL.device_create_command_encoder(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_get_queue(device_id: id::DeviceId) -> id::QueueId {
    device_id
}

#[no_mangle]
pub extern "C" fn wgpu_queue_submit(
    queue_id: id::QueueId,
    command_buffers: *const id::CommandBufferId,
    command_buffers_length: usize,
) {
    let command_buffer_ids =
        unsafe { slice::from_raw_parts(command_buffers, command_buffers_length) };
    gfx_select!(queue_id => GLOBAL.queue_submit(queue_id, command_buffer_ids))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_render_pipeline(
    device_id: id::DeviceId,
    desc: &core::pipeline::RenderPipelineDescriptor,
) -> id::RenderPipelineId {
    gfx_select!(device_id => GLOBAL.device_create_render_pipeline(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_compute_pipeline(
    device_id: id::DeviceId,
    desc: &core::pipeline::ComputePipelineDescriptor,
) -> id::ComputePipelineId {
    gfx_select!(device_id => GLOBAL.device_create_compute_pipeline(device_id, desc, PhantomData))
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_swap_chain(
    device_id: id::DeviceId,
    surface_id: id::SurfaceId,
    desc: &core::swap_chain::SwapChainDescriptor,
) -> id::SwapChainId {
    gfx_select!(device_id => GLOBAL.device_create_swap_chain(device_id, surface_id, desc))
}

#[no_mangle]
pub extern "C" fn wgpu_device_poll(device_id: id::DeviceId, force_wait: bool) {
    gfx_select!(device_id => GLOBAL.device_poll(device_id, force_wait))
}

#[no_mangle]
pub extern "C" fn wgpu_device_destroy(device_id: id::DeviceId) {
    gfx_select!(device_id => GLOBAL.device_destroy(device_id))
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_read_async(
    buffer_id: id::BufferId,
    start: core::BufferAddress,
    size: core::BufferAddress,
    callback: BufferMapReadCallback,
    userdata: *mut u8,
) {
    let operation = core::resource::BufferMapOperation::Read(
        start .. start + size,
        Box::new(move |status, data| unsafe {
            callback(status, data, userdata)
        }),
    );
    gfx_select!(buffer_id => GLOBAL.buffer_map_async(buffer_id, core::resource::BufferUsage::MAP_READ, operation))
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_write_async(
    buffer_id: id::BufferId,
    start: core::BufferAddress,
    size: core::BufferAddress,
    callback: BufferMapWriteCallback,
    userdata: *mut u8,
) {
    let operation = core::resource::BufferMapOperation::Write(
        start .. start + size,
        Box::new(move |status, data| unsafe {
            callback(status, data, userdata)
        }),
    );
    gfx_select!(buffer_id => GLOBAL.buffer_map_async(buffer_id, core::resource::BufferUsage::MAP_WRITE, operation))
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_unmap(buffer_id: id::BufferId) {
    gfx_select!(buffer_id => GLOBAL.buffer_unmap(buffer_id))
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_get_next_texture(
    swap_chain_id: id::SwapChainId,
) -> core::swap_chain::SwapChainOutput {
    gfx_select!(swap_chain_id => GLOBAL.swap_chain_get_next_texture(swap_chain_id, PhantomData))
        .unwrap_or(core::swap_chain::SwapChainOutput {
            view_id: id::TextureViewId::ERROR,
        })
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_present(swap_chain_id: id::SwapChainId) {
    gfx_select!(swap_chain_id => GLOBAL.swap_chain_present(swap_chain_id))
}
