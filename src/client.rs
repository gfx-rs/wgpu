/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    cow_label, ByteBuf, CommandEncoderAction, DeviceAction, DropAction, ImplicitLayout, RawString,
    TextureAction,
};

use wgc::{hub::IdentityManager, id};
use wgt::Backend;

pub use wgc::command::{compute_ffi::*, render_ffi::*};

use parking_lot::Mutex;

use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU8},
    ptr, slice,
};

fn make_byte_buf<T: serde::Serialize>(data: &T) -> ByteBuf {
    let vec = bincode::serialize(data).unwrap();
    ByteBuf::from_vec(vec)
}

#[repr(C)]
pub struct ShaderModuleDescriptor {
    spirv_words: *const u32,
    spirv_words_length: usize,
    wgsl_chars: RawString,
}

#[repr(C)]
pub struct ProgrammableStageDescriptor {
    module: id::ShaderModuleId,
    entry_point: RawString,
}

impl ProgrammableStageDescriptor {
    fn to_wgpu(&self) -> wgc::pipeline::ProgrammableStageDescriptor {
        wgc::pipeline::ProgrammableStageDescriptor {
            module: self.module,
            entry_point: cow_label(&self.entry_point).unwrap(),
        }
    }
}

#[repr(C)]
pub struct ComputePipelineDescriptor {
    label: RawString,
    layout: Option<id::PipelineLayoutId>,
    compute_stage: ProgrammableStageDescriptor,
}

#[repr(C)]
pub struct VertexBufferDescriptor {
    stride: wgt::BufferAddress,
    step_mode: wgt::InputStepMode,
    attributes: *const wgt::VertexAttributeDescriptor,
    attributes_length: usize,
}

#[repr(C)]
pub struct VertexStateDescriptor {
    index_format: wgt::IndexFormat,
    vertex_buffers: *const VertexBufferDescriptor,
    vertex_buffers_length: usize,
}

#[repr(C)]
pub struct RenderPipelineDescriptor<'a> {
    label: RawString,
    layout: Option<id::PipelineLayoutId>,
    vertex_stage: &'a ProgrammableStageDescriptor,
    fragment_stage: Option<&'a ProgrammableStageDescriptor>,
    primitive_topology: wgt::PrimitiveTopology,
    rasterization_state: Option<&'a wgt::RasterizationStateDescriptor>,
    color_states: *const wgt::ColorStateDescriptor,
    color_states_length: usize,
    depth_stencil_state: Option<&'a wgt::DepthStencilStateDescriptor>,
    vertex_state: VertexStateDescriptor,
    sample_count: u32,
    sample_mask: u32,
    alpha_to_coverage_enabled: bool,
}

#[repr(C)]
pub enum RawTextureSampleType {
    Float,
    UnfilterableFloat,
    Uint,
    Sint,
    Depth,
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

#[repr(C)]
pub struct BindGroupLayoutEntry<'a> {
    binding: u32,
    visibility: wgt::ShaderStage,
    ty: RawBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<wgt::BufferSize>,
    view_dimension: Option<&'a wgt::TextureViewDimension>,
    texture_sample_type: Option<&'a RawTextureSampleType>,
    multisampled: bool,
    storage_texture_format: Option<&'a wgt::TextureFormat>,
}

#[repr(C)]
pub struct BindGroupLayoutDescriptor<'a> {
    label: RawString,
    entries: *const BindGroupLayoutEntry<'a>,
    entries_length: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupEntry {
    binding: u32,
    buffer: Option<id::BufferId>,
    offset: wgt::BufferAddress,
    size: Option<wgt::BufferSize>,
    sampler: Option<id::SamplerId>,
    texture_view: Option<id::TextureViewId>,
}

#[repr(C)]
pub struct BindGroupDescriptor {
    label: RawString,
    layout: id::BindGroupLayoutId,
    entries: *const BindGroupEntry,
    entries_length: usize,
}

#[repr(C)]
pub struct PipelineLayoutDescriptor {
    label: RawString,
    bind_group_layouts: *const id::BindGroupLayoutId,
    bind_group_layouts_length: usize,
}

#[repr(C)]
pub struct SamplerDescriptor<'a> {
    label: RawString,
    address_modes: [wgt::AddressMode; 3],
    mag_filter: wgt::FilterMode,
    min_filter: wgt::FilterMode,
    mipmap_filter: wgt::FilterMode,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    compare: Option<&'a wgt::CompareFunction>,
    anisotropy_clamp: Option<NonZeroU8>,
}

#[repr(C)]
pub struct TextureViewDescriptor<'a> {
    label: RawString,
    format: Option<&'a wgt::TextureFormat>,
    dimension: Option<&'a wgt::TextureViewDimension>,
    aspect: wgt::TextureAspect,
    base_mip_level: u32,
    level_count: Option<NonZeroU32>,
    base_array_layer: u32,
    array_layer_count: Option<NonZeroU32>,
}

#[derive(Debug, Default)]
struct IdentityHub {
    adapters: IdentityManager,
    devices: IdentityManager,
    buffers: IdentityManager,
    command_buffers: IdentityManager,
    render_bundles: IdentityManager,
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

impl ImplicitLayout<'_> {
    fn new(identities: &mut IdentityHub, backend: Backend) -> Self {
        ImplicitLayout {
            pipeline: identities.pipeline_layouts.alloc(backend),
            bind_groups: Cow::Owned(
                (0..wgc::MAX_BIND_GROUPS)
                    .map(|_| identities.bind_group_layouts.alloc(backend))
                    .collect(),
            ),
        }
    }
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

#[no_mangle]
pub unsafe extern "C" fn wgpu_client_drop_action(client: &mut Client, byte_buf: &ByteBuf) {
    let mut cursor = std::io::Cursor::new(byte_buf.as_slice());
    let mut identities = client.identities.lock();
    while let Ok(action) = bincode::deserialize_from(&mut cursor) {
        match action {
            DropAction::Buffer(id) => identities.select(id.backend()).buffers.free(id),
            DropAction::Texture(id) => identities.select(id.backend()).textures.free(id),
            DropAction::Sampler(id) => identities.select(id.backend()).samplers.free(id),
            DropAction::BindGroupLayout(id) => {
                identities.select(id.backend()).bind_group_layouts.free(id)
            }
        }
    }
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
pub extern "C" fn wgpu_client_create_buffer(
    client: &Client,
    device_id: id::DeviceId,
    desc: &wgt::BufferDescriptor<RawString>,
    bb: &mut ByteBuf,
) -> id::BufferId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .buffers
        .alloc(backend);

    let action = DeviceAction::CreateBuffer(id, desc.map_label(cow_label));
    *bb = make_byte_buf(&action);
    id
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
pub extern "C" fn wgpu_client_create_texture(
    client: &Client,
    device_id: id::DeviceId,
    desc: &wgt::TextureDescriptor<RawString>,
    bb: &mut ByteBuf,
) -> id::TextureId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .textures
        .alloc(backend);

    let action = DeviceAction::CreateTexture(id, desc.map_label(cow_label));
    *bb = make_byte_buf(&action);
    id
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
pub extern "C" fn wgpu_client_create_texture_view(
    client: &Client,
    device_id: id::DeviceId,
    desc: &TextureViewDescriptor,
    bb: &mut ByteBuf,
) -> id::TextureViewId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .texture_views
        .alloc(backend);

    let wgpu_desc = wgc::resource::TextureViewDescriptor {
        label: cow_label(&desc.label),
        format: desc.format.cloned(),
        dimension: desc.dimension.cloned(),
        aspect: desc.aspect,
        base_mip_level: desc.base_mip_level,
        level_count: desc.level_count,
        base_array_layer: desc.base_array_layer,
        array_layer_count: desc.array_layer_count,
    };

    let action = TextureAction::CreateView(id, wgpu_desc);
    *bb = make_byte_buf(&action);
    id
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
pub extern "C" fn wgpu_client_create_sampler(
    client: &Client,
    device_id: id::DeviceId,
    desc: &SamplerDescriptor,
    bb: &mut ByteBuf,
) -> id::SamplerId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .samplers
        .alloc(backend);

    let wgpu_desc = wgc::resource::SamplerDescriptor {
        label: cow_label(&desc.label),
        address_modes: desc.address_modes,
        mag_filter: desc.mag_filter,
        min_filter: desc.min_filter,
        mipmap_filter: desc.mipmap_filter,
        lod_min_clamp: desc.lod_min_clamp,
        lod_max_clamp: desc.lod_max_clamp,
        compare: desc.compare.cloned(),
        anisotropy_clamp: desc.anisotropy_clamp,
        border_color: None,
    };
    let action = DeviceAction::CreateSampler(id, wgpu_desc);
    *bb = make_byte_buf(&action);
    id
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
pub extern "C" fn wgpu_client_create_command_encoder(
    client: &Client,
    device_id: id::DeviceId,
    desc: &wgt::CommandEncoderDescriptor<RawString>,
    bb: &mut ByteBuf,
) -> id::CommandEncoderId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .command_buffers
        .alloc(backend);

    let action = DeviceAction::CreateCommandEncoder(id, desc.map_label(cow_label));
    *bb = make_byte_buf(&action);
    id
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
) -> *mut wgc::command::ComputePass {
    let pass = wgc::command::ComputePass::new(encoder_id);
    Box::into_raw(Box::new(pass))
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_compute_pass_finish(
    pass: *mut wgc::command::ComputePass,
    output: &mut ByteBuf,
) {
    let command = Box::from_raw(pass).into_command();
    *output = make_byte_buf(&command);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_compute_pass_destroy(pass: *mut wgc::command::ComputePass) {
    let _ = Box::from_raw(pass);
}

#[repr(C)]
pub struct RenderPassDescriptor {
    pub color_attachments: *const wgc::command::ColorAttachmentDescriptor,
    pub color_attachments_length: usize,
    pub depth_stencil_attachment: *const wgc::command::DepthStencilAttachmentDescriptor,
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_render_pass(
    encoder_id: id::CommandEncoderId,
    desc: &RenderPassDescriptor,
) -> *mut wgc::command::RenderPass {
    let pass = wgc::command::RenderPass::new(
        encoder_id,
        wgc::command::RenderPassDescriptor {
            color_attachments: Cow::Borrowed(slice::from_raw_parts(
                desc.color_attachments,
                desc.color_attachments_length,
            )),
            depth_stencil_attachment: desc.depth_stencil_attachment.as_ref(),
        },
    );
    Box::into_raw(Box::new(pass))
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_render_pass_finish(
    pass: *mut wgc::command::RenderPass,
    output: &mut ByteBuf,
) {
    let command = Box::from_raw(pass).into_command();
    *output = make_byte_buf(&command);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_render_pass_destroy(pass: *mut wgc::command::RenderPass) {
    let _ = Box::from_raw(pass);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_client_create_bind_group_layout(
    client: &Client,
    device_id: id::DeviceId,
    desc: &BindGroupLayoutDescriptor,
    bb: &mut ByteBuf,
) -> id::BindGroupLayoutId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .bind_group_layouts
        .alloc(backend);

    let mut entries = Vec::with_capacity(desc.entries_length);
    for entry in slice::from_raw_parts(desc.entries, desc.entries_length) {
        entries.push(wgt::BindGroupLayoutEntry {
            binding: entry.binding,
            visibility: entry.visibility,
            count: None,
            ty: match entry.ty {
                RawBindingType::UniformBuffer => wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: entry.has_dynamic_offset,
                    min_binding_size: entry.min_binding_size,
                },
                RawBindingType::StorageBuffer => wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: entry.has_dynamic_offset,
                    min_binding_size: entry.min_binding_size,
                },
                RawBindingType::ReadonlyStorageBuffer => wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: entry.has_dynamic_offset,
                    min_binding_size: entry.min_binding_size,
                },
                RawBindingType::Sampler => wgt::BindingType::Sampler {
                    comparison: false,
                    filtering: false,
                },
                RawBindingType::ComparisonSampler => wgt::BindingType::Sampler {
                    comparison: true,
                    filtering: false,
                },
                RawBindingType::SampledTexture => wgt::BindingType::Texture {
                    //TODO: the spec has a bug here
                    view_dimension: *entry
                        .view_dimension
                        .unwrap_or(&wgt::TextureViewDimension::D2),
                    sample_type: match entry.texture_sample_type {
                        None | Some(RawTextureSampleType::Float) => {
                            wgt::TextureSampleType::Float { filterable: true }
                        }
                        Some(RawTextureSampleType::UnfilterableFloat) => {
                            wgt::TextureSampleType::Float { filterable: false }
                        }
                        Some(RawTextureSampleType::Uint) => wgt::TextureSampleType::Uint,
                        Some(RawTextureSampleType::Sint) => wgt::TextureSampleType::Sint,
                        Some(RawTextureSampleType::Depth) => wgt::TextureSampleType::Depth,
                    },
                    multisampled: entry.multisampled,
                },
                RawBindingType::ReadonlyStorageTexture => wgt::BindingType::StorageTexture {
                    access: wgt::StorageTextureAccess::ReadOnly,
                    view_dimension: *entry.view_dimension.unwrap(),
                    format: *entry.storage_texture_format.unwrap(),
                },
                RawBindingType::WriteonlyStorageTexture => wgt::BindingType::StorageTexture {
                    access: wgt::StorageTextureAccess::WriteOnly,
                    view_dimension: *entry.view_dimension.unwrap(),
                    format: *entry.storage_texture_format.unwrap(),
                },
            },
        });
    }
    let wgpu_desc = wgc::binding_model::BindGroupLayoutDescriptor {
        label: cow_label(&desc.label),
        entries: Cow::Owned(entries),
    };

    let action = DeviceAction::CreateBindGroupLayout(id, wgpu_desc);
    *bb = make_byte_buf(&action);
    id
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
pub unsafe extern "C" fn wgpu_client_create_pipeline_layout(
    client: &Client,
    device_id: id::DeviceId,
    desc: &PipelineLayoutDescriptor,
    bb: &mut ByteBuf,
) -> id::PipelineLayoutId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .pipeline_layouts
        .alloc(backend);

    let wgpu_desc = wgc::binding_model::PipelineLayoutDescriptor {
        label: cow_label(&desc.label),
        bind_group_layouts: Cow::Borrowed(slice::from_raw_parts(
            desc.bind_group_layouts,
            desc.bind_group_layouts_length,
        )),
        push_constant_ranges: Cow::Borrowed(&[]),
    };

    let action = DeviceAction::CreatePipelineLayout(id, wgpu_desc);
    *bb = make_byte_buf(&action);
    id
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
pub unsafe extern "C" fn wgpu_client_create_bind_group(
    client: &Client,
    device_id: id::DeviceId,
    desc: &BindGroupDescriptor,
    bb: &mut ByteBuf,
) -> id::BindGroupId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .bind_groups
        .alloc(backend);

    let mut entries = Vec::with_capacity(desc.entries_length);
    for entry in slice::from_raw_parts(desc.entries, desc.entries_length) {
        entries.push(wgc::binding_model::BindGroupEntry {
            binding: entry.binding,
            resource: if let Some(id) = entry.buffer {
                wgc::binding_model::BindingResource::Buffer(wgc::binding_model::BufferBinding {
                    buffer_id: id,
                    offset: entry.offset,
                    size: entry.size,
                })
            } else if let Some(id) = entry.sampler {
                wgc::binding_model::BindingResource::Sampler(id)
            } else if let Some(id) = entry.texture_view {
                wgc::binding_model::BindingResource::TextureView(id)
            } else {
                panic!("Unexpected binding entry {:?}", entry);
            },
        });
    }
    let wgpu_desc = wgc::binding_model::BindGroupDescriptor {
        label: cow_label(&desc.label),
        layout: desc.layout,
        entries: Cow::Owned(entries),
    };

    let action = DeviceAction::CreateBindGroup(id, wgpu_desc);
    *bb = make_byte_buf(&action);
    id
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
pub unsafe extern "C" fn wgpu_client_create_shader_module(
    client: &Client,
    device_id: id::DeviceId,
    desc: &ShaderModuleDescriptor,
    bb: &mut ByteBuf,
) -> id::ShaderModuleId {
    let backend = device_id.backend();
    let id = client
        .identities
        .lock()
        .select(backend)
        .shader_modules
        .alloc(backend);

    assert!(!desc.spirv_words.is_null());
    let spv = Cow::Borrowed(if desc.spirv_words.is_null() {
        &[][..]
    } else {
        slice::from_raw_parts(desc.spirv_words, desc.spirv_words_length)
    });

    let wgsl = cow_label(&desc.wgsl_chars).unwrap_or_default();

    let action = DeviceAction::CreateShaderModule(id, spv, wgsl);
    *bb = make_byte_buf(&action);
    id
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
pub unsafe extern "C" fn wgpu_client_create_compute_pipeline(
    client: &Client,
    device_id: id::DeviceId,
    desc: &ComputePipelineDescriptor,
    bb: &mut ByteBuf,
    implicit_bind_group_layout_ids: *mut Option<id::BindGroupLayoutId>,
) -> id::ComputePipelineId {
    let backend = device_id.backend();
    let mut identities = client.identities.lock();
    let id = identities.select(backend).compute_pipelines.alloc(backend);

    let wgpu_desc = wgc::pipeline::ComputePipelineDescriptor {
        label: cow_label(&desc.label),
        layout: desc.layout,
        compute_stage: desc.compute_stage.to_wgpu(),
    };

    let implicit = match desc.layout {
        Some(_) => None,
        None => {
            let implicit = ImplicitLayout::new(identities.select(backend), backend);
            for (i, bgl_id) in implicit.bind_groups.iter().enumerate() {
                *implicit_bind_group_layout_ids.add(i) = Some(*bgl_id);
            }
            Some(implicit)
        }
    };

    let action = DeviceAction::CreateComputePipeline(id, wgpu_desc, implicit);
    *bb = make_byte_buf(&action);
    id
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
pub unsafe extern "C" fn wgpu_client_create_render_pipeline(
    client: &Client,
    device_id: id::DeviceId,
    desc: &RenderPipelineDescriptor,
    bb: &mut ByteBuf,
    implicit_bind_group_layout_ids: *mut Option<id::BindGroupLayoutId>,
) -> id::RenderPipelineId {
    let backend = device_id.backend();
    let mut identities = client.identities.lock();
    let id = identities.select(backend).render_pipelines.alloc(backend);

    let wgpu_desc = wgc::pipeline::RenderPipelineDescriptor {
        label: cow_label(&desc.label),
        layout: desc.layout,
        vertex_stage: desc.vertex_stage.to_wgpu(),
        fragment_stage: desc
            .fragment_stage
            .map(ProgrammableStageDescriptor::to_wgpu),
        rasterization_state: desc.rasterization_state.cloned(),
        primitive_topology: desc.primitive_topology,
        color_states: Cow::Borrowed(slice::from_raw_parts(
            desc.color_states,
            desc.color_states_length,
        )),
        depth_stencil_state: desc.depth_stencil_state.cloned(),
        vertex_state: wgc::pipeline::VertexStateDescriptor {
            index_format: desc.vertex_state.index_format,
            vertex_buffers: {
                let vbufs = slice::from_raw_parts(
                    desc.vertex_state.vertex_buffers,
                    desc.vertex_state.vertex_buffers_length,
                );
                let owned = vbufs
                    .iter()
                    .map(|vb| wgc::pipeline::VertexBufferDescriptor {
                        stride: vb.stride,
                        step_mode: vb.step_mode,
                        attributes: Cow::Borrowed(if vb.attributes.is_null() {
                            &[]
                        } else {
                            slice::from_raw_parts(vb.attributes, vb.attributes_length)
                        }),
                    })
                    .collect();
                Cow::Owned(owned)
            },
        },
        sample_count: desc.sample_count,
        sample_mask: desc.sample_mask,
        alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
    };

    let implicit = match desc.layout {
        Some(_) => None,
        None => {
            let implicit = ImplicitLayout::new(identities.select(backend), backend);
            for (i, bgl_id) in implicit.bind_groups.iter().enumerate() {
                *implicit_bind_group_layout_ids.add(i) = Some(*bgl_id);
            }
            Some(implicit)
        }
    };

    let action = DeviceAction::CreateRenderPipeline(id, wgpu_desc, implicit);
    *bb = make_byte_buf(&action);
    id
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

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_copy_buffer_to_buffer(
    src: id::BufferId,
    src_offset: wgt::BufferAddress,
    dst: id::BufferId,
    dst_offset: wgt::BufferAddress,
    size: wgt::BufferAddress,
    bb: &mut ByteBuf,
) {
    let action = CommandEncoderAction::CopyBufferToBuffer {
        src,
        src_offset,
        dst,
        dst_offset,
        size,
    };
    *bb = make_byte_buf(&action);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_copy_texture_to_buffer(
    src: wgc::command::TextureCopyView,
    dst: wgc::command::BufferCopyView,
    size: wgt::Extent3d,
    bb: &mut ByteBuf,
) {
    let action = CommandEncoderAction::CopyTextureToBuffer { src, dst, size };
    *bb = make_byte_buf(&action);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_copy_buffer_to_texture(
    src: wgc::command::BufferCopyView,
    dst: wgc::command::TextureCopyView,
    size: wgt::Extent3d,
    bb: &mut ByteBuf,
) {
    let action = CommandEncoderAction::CopyBufferToTexture { src, dst, size };
    *bb = make_byte_buf(&action);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_copy_texture_to_texture(
    src: wgc::command::TextureCopyView,
    dst: wgc::command::TextureCopyView,
    size: wgt::Extent3d,
    bb: &mut ByteBuf,
) {
    let action = CommandEncoderAction::CopyTextureToTexture { src, dst, size };
    *bb = make_byte_buf(&action);
}
