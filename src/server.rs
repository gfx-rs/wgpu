/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::identity::IdentityRecyclerFactory;

use wgc::{gfx_select, id};

use std::{marker::PhantomData, mem, slice};

// hide wgc's global in private
pub struct Global(wgc::hub::Global<IdentityRecyclerFactory>);
pub type RawString = *const std::os::raw::c_char;

impl std::ops::Deref for Global {
    type Target = wgc::hub::Global<IdentityRecyclerFactory>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(C)]
pub enum RawBindingType {
    UniformBuffer,
    StorageBuffer,
    ReadonlyStorageBuffer,
    Sampler,
    ComparisonSampler,
    SampledTexture,
    ReadonlyStorageTexture,
    WriteonlyStorageTexture,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct RawEnumOption<T>(u32, PhantomData<T>);

impl<T: Copy> From<Option<T>> for RawEnumOption<T> {
    fn from(option: Option<T>) -> Self {
        debug_assert_eq!(mem::size_of::<T>(), 4);
        let value = match option {
            Some(ref v) => unsafe { *mem::transmute::<*const T, *const u32>(v) },
            None => !0,
        };
        RawEnumOption(value, PhantomData)
    }
}

impl<T: Copy> RawEnumOption<T> {
    fn unwrap(self) -> T {
        assert_ne!(self.0, !0);
        unsafe { *mem::transmute::<*const u32, *const T>(&self.0) }
    }
}

#[repr(C)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: wgt::ShaderStage,
    pub ty: RawBindingType,
    pub has_dynamic_offset: bool,
    pub view_dimension: RawEnumOption<wgt::TextureViewDimension>,
    pub texture_component_type: RawEnumOption<wgt::TextureComponentType>,
    pub multisampled: bool,
    pub storage_texture_format: RawEnumOption<wgt::TextureFormat>,
}

#[repr(C)]
pub struct BindGroupLayoutDescriptor {
    pub label: RawString,
    pub entries: *const BindGroupLayoutEntry,
    pub entries_length: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub buffer: Option<id::BufferId>,
    pub offset: wgt::BufferAddress,
    pub size: wgt::BufferSize,
    pub sampler: Option<id::SamplerId>,
    pub texture_view: Option<id::TextureViewId>,
}

#[repr(C)]
pub struct BindGroupDescriptor {
    pub label: RawString,
    pub layout: id::BindGroupLayoutId,
    pub entries: *const BindGroupEntry,
    pub entries_length: usize,
}

#[repr(C)]
pub struct SamplerDescriptor<'a> {
    pub label: RawString,
    pub address_modes: [wgt::AddressMode; 3],
    pub mag_filter: wgt::FilterMode,
    pub min_filter: wgt::FilterMode,
    pub mipmap_filter: wgt::FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<&'a wgt::CompareFunction>,
    pub anisotropy_clamp: u8,
}

#[no_mangle]
pub extern "C" fn wgpu_server_new(factory: IdentityRecyclerFactory) -> *mut Global {
    log::info!("Initializing WGPU server");
    let global = Global(wgc::hub::Global::new("wgpu", factory));
    Box::into_raw(Box::new(global))
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_delete(global: *mut Global) {
    log::info!("Terminating WGPU server");
    let _ = Box::from_raw(global);
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
        wgt::UnsafeExtensions::disallow(),
        wgc::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Some(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        None => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_adapter_request_device(
    global: &Global,
    self_id: id::AdapterId,
    desc: &wgt::DeviceDescriptor,
    new_id: id::DeviceId,
) {
    let trace_string = std::env::var("WGPU_TRACE").ok();
    let trace_path = trace_string
        .as_ref()
        .map(|string| std::path::Path::new(string.as_str()));
    gfx_select!(self_id => global.adapter_request_device(self_id, desc, trace_path, new_id));
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
    desc: &wgt::BufferDescriptor<RawString>,
    new_id: id::BufferId,
) {
    gfx_select!(self_id => global.device_create_buffer(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_map(
    global: &Global,
    buffer_id: id::BufferId,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
    operation: wgc::resource::BufferMapOperation,
) {
    gfx_select!(buffer_id => global.buffer_map_async(
        buffer_id,
        start .. start + size,
        operation
    ));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `size` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_buffer_get_mapped_range(
    global: &Global,
    buffer_id: id::BufferId,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
) -> *mut u8 {
    gfx_select!(buffer_id => global.buffer_get_mapped_range(
        buffer_id,
        start,
        wgt::BufferSize(size)
    ))
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
    size: &wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_texture_to_buffer(self_id, source, destination, size));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_buffer_to_texture(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::BufferCopyView,
    destination: &wgc::command::TextureCopyView,
    size: &wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_buffer_to_texture(self_id, source, destination, size));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_texture_to_texture(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::TextureCopyView,
    destination: &wgc::command::TextureCopyView,
    size: &wgt::Extent3d,
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

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `data_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_write_buffer(
    global: &Global,
    self_id: id::QueueId,
    buffer_id: id::BufferId,
    buffer_offset: wgt::BufferAddress,
    data: *const u8,
    data_length: usize,
) {
    let data = slice::from_raw_parts(data, data_length);
    gfx_select!(self_id => global.queue_write_buffer(self_id, buffer_id, buffer_offset, data));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `data_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_write_texture(
    global: &Global,
    self_id: id::QueueId,
    destination: &wgc::command::TextureCopyView,
    data: *const u8,
    data_length: usize,
    layout: &wgt::TextureDataLayout,
    extent: &wgt::Extent3d,
) {
    let data = slice::from_raw_parts(data, data_length);
    gfx_select!(self_id => global.queue_write_texture(self_id, destination, data, layout, extent));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `entries_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_device_create_bind_group_layout(
    global: &Global,
    self_id: id::DeviceId,
    desc: &BindGroupLayoutDescriptor,
    new_id: id::BindGroupLayoutId,
) {
    let entries = slice::from_raw_parts(desc.entries, desc.entries_length);
    let bindings = entries
        .iter()
        .map(|entry| wgt::BindGroupLayoutEntry {
            binding: entry.binding,
            visibility: entry.visibility,
            ty: match entry.ty {
                RawBindingType::UniformBuffer => wgt::BindingType::UniformBuffer {
                    dynamic: entry.has_dynamic_offset,
                },
                RawBindingType::StorageBuffer => wgt::BindingType::StorageBuffer {
                    dynamic: entry.has_dynamic_offset,
                    readonly: false,
                },
                RawBindingType::ReadonlyStorageBuffer => wgt::BindingType::StorageBuffer {
                    dynamic: entry.has_dynamic_offset,
                    readonly: true,
                },
                RawBindingType::Sampler => wgt::BindingType::Sampler { comparison: false },
                RawBindingType::ComparisonSampler => wgt::BindingType::Sampler { comparison: true },
                RawBindingType::SampledTexture => wgt::BindingType::SampledTexture {
                    dimension: entry.view_dimension.unwrap(),
                    component_type: entry.texture_component_type.unwrap(),
                    multisampled: entry.multisampled,
                },
                RawBindingType::ReadonlyStorageTexture => wgt::BindingType::StorageTexture {
                    dimension: entry.view_dimension.unwrap(),
                    component_type: entry.texture_component_type.unwrap(),
                    format: entry.storage_texture_format.unwrap(),
                    readonly: true,
                },
                RawBindingType::WriteonlyStorageTexture => wgt::BindingType::StorageTexture {
                    dimension: entry.view_dimension.unwrap(),
                    component_type: entry.texture_component_type.unwrap(),
                    format: entry.storage_texture_format.unwrap(),
                    readonly: false,
                },
            },
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let desc = wgt::BindGroupLayoutDescriptor {
        label: None,
        bindings: &bindings,
    };
    gfx_select!(self_id => global.device_create_bind_group_layout(self_id, &desc, new_id)).unwrap();
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
    gfx_select!(self_id => global.device_create_pipeline_layout(self_id, desc, new_id)).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_pipeline_layout_destroy(
    global: &Global,
    self_id: id::PipelineLayoutId,
) {
    gfx_select!(self_id => global.pipeline_layout_destroy(self_id));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `entries_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_device_create_bind_group(
    global: &Global,
    self_id: id::DeviceId,
    desc: &BindGroupDescriptor,
    new_id: id::BindGroupId,
) {
    let entries = slice::from_raw_parts(desc.entries, desc.entries_length);
    let bindings = entries
        .iter()
        .map(|entry| wgc::binding_model::BindGroupEntry {
            binding: entry.binding,
            resource: if let Some(id) = entry.buffer {
                wgc::binding_model::BindingResource::Buffer(wgc::binding_model::BufferBinding {
                    buffer: id,
                    offset: entry.offset,
                    size: entry.size,
                })
            } else if let Some(id) = entry.sampler {
                wgc::binding_model::BindingResource::Sampler(id)
            } else if let Some(id) = entry.texture_view {
                wgc::binding_model::BindingResource::TextureView(id)
            } else {
                panic!("Unrecognized binding entry: {:?}", entry);
            },
        })
        .collect::<Vec<_>>();
    let desc = wgc::binding_model::BindGroupDescriptor {
        label: None,
        layout: desc.layout,
        bindings: &bindings,
    };
    gfx_select!(self_id => global.device_create_bind_group(self_id, &desc, new_id));
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
    gfx_select!(self_id => global.device_create_compute_pipeline(self_id, desc, new_id)).unwrap();
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
    gfx_select!(self_id => global.device_create_render_pipeline(self_id, desc, new_id)).unwrap();
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
    desc: &wgt::TextureDescriptor<RawString>,
    new_id: id::TextureId,
) {
    gfx_select!(self_id => global.device_create_texture(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_create_view(
    global: &Global,
    self_id: id::TextureId,
    desc: Option<&wgt::TextureViewDescriptor<RawString>>,
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
    desc: &SamplerDescriptor,
    new_id: id::SamplerId,
) {
    let desc = wgt::SamplerDescriptor {
        label: desc.label,
        address_mode_u: desc.address_modes[0],
        address_mode_v: desc.address_modes[1],
        address_mode_w: desc.address_modes[2],
        mag_filter: desc.mag_filter,
        min_filter: desc.min_filter,
        mipmap_filter: desc.mipmap_filter,
        lod_min_clamp: desc.lod_min_clamp,
        lod_max_clamp: desc.lod_max_clamp,
        compare: desc.compare.cloned(),
        anisotropy_clamp: if desc.anisotropy_clamp > 1 {
            Some(desc.anisotropy_clamp)
        } else {
            None
        },
        _non_exhaustive: unsafe { wgt::NonExhaustive::new() },
    };
    gfx_select!(self_id => global.device_create_sampler(self_id, &desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_sampler_destroy(global: &Global, self_id: id::SamplerId) {
    gfx_select!(self_id => global.sampler_destroy(self_id));
}
