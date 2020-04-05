use wgn;

use crate::{
    backend::native_gpu_future, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource,
    BindingType, BufferDescriptor, CommandEncoderDescriptor, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, TextureDescriptor,
    TextureViewDimension,
};

use arrayvec::ArrayVec;
use smallvec::SmallVec;
use std::{ffi::CString, future::Future, ops::Range, ptr, slice};

pub type AdapterId = wgc::id::AdapterId;
pub type DeviceId = wgc::id::DeviceId;
pub type QueueId = wgc::id::QueueId;
pub type ShaderModuleId = wgc::id::ShaderModuleId;
pub type BindGroupLayoutId = wgc::id::BindGroupLayoutId;
pub type BindGroupId = wgc::id::BindGroupId;
pub type TextureViewId = wgc::id::TextureViewId;
pub type SamplerId = wgc::id::SamplerId;
pub type BufferId = wgc::id::BufferId;
pub type TextureId = wgc::id::TextureId;
pub type PipelineLayoutId = wgc::id::PipelineLayoutId;
pub type RenderPipelineId = wgc::id::RenderPipelineId;
pub type ComputePipelineId = wgc::id::ComputePipelineId;
pub type CommandEncoderId = wgc::id::CommandEncoderId;
pub type ComputePassId = wgc::id::ComputePassId;
pub type CommandBufferId = wgc::id::CommandBufferId;
pub type SurfaceId = wgc::id::SurfaceId;
pub type SwapChainId = wgc::id::SwapChainId;
pub type RenderPassEncoderId = wgc::id::RenderPassId;

fn map_buffer_copy_view(view: crate::BufferCopyView<'_>) -> wgc::command::BufferCopyView {
    wgc::command::BufferCopyView {
        buffer: view.buffer.id,
        offset: view.offset,
        bytes_per_row: view.bytes_per_row,
        rows_per_image: view.rows_per_image,
    }
}

fn map_texture_copy_view<'a>(view: crate::TextureCopyView<'a>) -> wgc::command::TextureCopyView {
    wgc::command::TextureCopyView {
        texture: view.texture.id,
        mip_level: view.mip_level,
        array_layer: view.array_layer,
        origin: view.origin,
    }
}

pub(crate) async fn request_adapter(
    options: &crate::RequestAdapterOptions<'_>,
    backends: wgt::BackendBit,
) -> Option<AdapterId> {
    unsafe extern "C" fn adapter_callback(
        id: Option<wgc::id::AdapterId>,
        user_data: *mut std::ffi::c_void,
    ) {
        *(user_data as *mut Option<wgc::id::AdapterId>) = id;
    }

    let mut id_maybe = None;
    unsafe {
        wgn::wgpu_request_adapter_async(
            Some(&wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: options.compatible_surface.map(|surface| surface.id),
            }),
            backends,
            adapter_callback,
            &mut id_maybe as *mut _ as *mut std::ffi::c_void,
        )
    };
    id_maybe
}

pub(crate) async fn request_device_and_queue(
    adapter: &AdapterId,
    desc: Option<&wgt::DeviceDescriptor>,
) -> (DeviceId, QueueId) {
    let device_id = wgn::wgpu_adapter_request_device(*adapter, desc);
    (device_id, wgn::wgpu_device_get_default_queue(device_id))
}

pub(crate) fn create_shader_module(device: &DeviceId, spv: &[u32]) -> ShaderModuleId {
    let desc = wgc::pipeline::ShaderModuleDescriptor {
        code: wgc::U32Array {
            bytes: spv.as_ptr(),
            length: spv.len(),
        },
    };
    wgn::wgpu_device_create_shader_module(*device, &desc)
}

pub(crate) fn create_bind_group_layout(
    device: &DeviceId,
    desc: &BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    use wgc::binding_model as bm;

    let temp_layouts = desc
        .bindings
        .iter()
        .map(|bind| bm::BindGroupLayoutEntry {
            binding: bind.binding,
            visibility: bind.visibility,
            ty: match bind.ty {
                BindingType::UniformBuffer { .. } => bm::BindingType::UniformBuffer,
                BindingType::StorageBuffer {
                    readonly: false, ..
                } => bm::BindingType::StorageBuffer,
                BindingType::StorageBuffer { readonly: true, .. } => {
                    bm::BindingType::ReadonlyStorageBuffer
                }
                BindingType::Sampler { comparison: false } => bm::BindingType::Sampler,
                BindingType::Sampler { .. } => bm::BindingType::ComparisonSampler,
                BindingType::SampledTexture { .. } => bm::BindingType::SampledTexture,
                BindingType::StorageTexture { readonly: true, .. } => {
                    bm::BindingType::ReadonlyStorageTexture
                }
                BindingType::StorageTexture { .. } => bm::BindingType::WriteonlyStorageTexture,
            },
            has_dynamic_offset: match bind.ty {
                BindingType::UniformBuffer { dynamic }
                | BindingType::StorageBuffer { dynamic, .. } => dynamic,
                _ => false,
            },
            multisampled: match bind.ty {
                BindingType::SampledTexture { multisampled, .. } => multisampled,
                _ => false,
            },
            view_dimension: match bind.ty {
                BindingType::SampledTexture { dimension, .. }
                | BindingType::StorageTexture { dimension, .. } => dimension,
                _ => TextureViewDimension::D2,
            },
            texture_component_type: match bind.ty {
                BindingType::SampledTexture { component_type, .. }
                | BindingType::StorageTexture { component_type, .. } => component_type,
                _ => wgt::TextureComponentType::Float,
            },
            storage_texture_format: match bind.ty {
                BindingType::StorageTexture { format, .. } => format,
                _ => wgt::TextureFormat::Rgb10a2Unorm, // doesn't matter
            },
        })
        .collect::<Vec<_>>();

    let owned_label = OwnedLabel::new(desc.label.as_deref());
    wgn::wgpu_device_create_bind_group_layout(
        *device,
        &bm::BindGroupLayoutDescriptor {
            entries: temp_layouts.as_ptr(),
            entries_length: temp_layouts.len(),
            label: owned_label.as_ptr(),
        },
    )
}

pub(crate) fn create_bind_group(device: &DeviceId, desc: &BindGroupDescriptor) -> BindGroupId {
    use wgc::binding_model as bm;

    let bindings = desc
        .bindings
        .iter()
        .map(|binding| bm::BindGroupEntry {
            binding: binding.binding,
            resource: match binding.resource {
                BindingResource::Buffer {
                    ref buffer,
                    ref range,
                } => bm::BindingResource::Buffer(bm::BufferBinding {
                    buffer: buffer.id,
                    offset: range.start,
                    size: range.end - range.start,
                }),
                BindingResource::Sampler(ref sampler) => bm::BindingResource::Sampler(sampler.id),
                BindingResource::TextureView(ref texture_view) => {
                    bm::BindingResource::TextureView(texture_view.id)
                }
            },
        })
        .collect::<Vec<_>>();

    let owned_label = OwnedLabel::new(desc.label.as_deref());
    wgn::wgpu_device_create_bind_group(
        *device,
        &bm::BindGroupDescriptor {
            layout: desc.layout.id,
            entries: bindings.as_ptr(),
            entries_length: bindings.len(),
            label: owned_label.as_ptr(),
        },
    )
}

pub(crate) fn create_pipeline_layout(
    device: &DeviceId,
    desc: &PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    //TODO: avoid allocation here
    let temp_layouts = desc
        .bind_group_layouts
        .iter()
        .map(|bgl| bgl.id)
        .collect::<Vec<_>>();
    wgn::wgpu_device_create_pipeline_layout(
        *device,
        &wgc::binding_model::PipelineLayoutDescriptor {
            bind_group_layouts: temp_layouts.as_ptr(),
            bind_group_layouts_length: temp_layouts.len(),
        },
    )
}

pub(crate) fn create_render_pipeline(
    device: &DeviceId,
    desc: &RenderPipelineDescriptor,
) -> RenderPipelineId {
    use wgc::pipeline as pipe;

    let vertex_entry_point = CString::new(desc.vertex_stage.entry_point).unwrap();
    let vertex_stage = pipe::ProgrammableStageDescriptor {
        module: desc.vertex_stage.module.id,
        entry_point: vertex_entry_point.as_ptr(),
    };
    let (_fragment_entry_point, fragment_stage) = if let Some(fragment_stage) = &desc.fragment_stage
    {
        let fragment_entry_point = CString::new(fragment_stage.entry_point).unwrap();
        let fragment_stage = pipe::ProgrammableStageDescriptor {
            module: fragment_stage.module.id,
            entry_point: fragment_entry_point.as_ptr(),
        };
        (fragment_entry_point, Some(fragment_stage))
    } else {
        (CString::default(), None)
    };

    let temp_color_states = desc.color_states.to_vec();
    let temp_vertex_buffers = desc
        .vertex_state
        .vertex_buffers
        .iter()
        .map(|vbuf| pipe::VertexBufferLayoutDescriptor {
            array_stride: vbuf.stride,
            step_mode: vbuf.step_mode,
            attributes: vbuf.attributes.as_ptr(),
            attributes_length: vbuf.attributes.len(),
        })
        .collect::<Vec<_>>();

    wgn::wgpu_device_create_render_pipeline(
        *device,
        &pipe::RenderPipelineDescriptor {
            layout: desc.layout.id,
            vertex_stage,
            fragment_stage: fragment_stage
                .as_ref()
                .map_or(ptr::null(), |fs| fs as *const _),
            rasterization_state: desc
                .rasterization_state
                .as_ref()
                .map_or(ptr::null(), |p| p as *const _),
            primitive_topology: desc.primitive_topology,
            color_states: temp_color_states.as_ptr(),
            color_states_length: temp_color_states.len(),
            depth_stencil_state: desc
                .depth_stencil_state
                .as_ref()
                .map_or(ptr::null(), |p| p as *const _),
            vertex_state: pipe::VertexStateDescriptor {
                index_format: desc.vertex_state.index_format,
                vertex_buffers: temp_vertex_buffers.as_ptr(),
                vertex_buffers_length: temp_vertex_buffers.len(),
            },
            sample_count: desc.sample_count,
            sample_mask: desc.sample_mask,
            alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
        },
    )
}

pub(crate) fn create_compute_pipeline(
    device: &DeviceId,
    desc: &ComputePipelineDescriptor,
) -> ComputePipelineId {
    use wgc::pipeline as pipe;

    let entry_point = CString::new(desc.compute_stage.entry_point).unwrap();

    wgn::wgpu_device_create_compute_pipeline(
        *device,
        &pipe::ComputePipelineDescriptor {
            layout: desc.layout.id,
            compute_stage: pipe::ProgrammableStageDescriptor {
                module: desc.compute_stage.module.id,
                entry_point: entry_point.as_ptr(),
            },
        },
    )
}

pub(crate) type CreateBufferMappedDetail = BufferDetail;

pub(crate) fn device_create_buffer_mapped<'a>(
    device: &DeviceId,
    desc: &BufferDescriptor,
) -> crate::CreateBufferMapped<'a> {
    let owned_label = OwnedLabel::new(desc.label.as_deref());
    let mut data_ptr: *mut u8 = std::ptr::null_mut();
    unsafe {
        let id = wgn::wgpu_device_create_buffer_mapped(
            *device,
            &wgt::BufferDescriptor {
                label: owned_label.as_ptr(),
                size: desc.size,
                usage: desc.usage,
            },
            &mut data_ptr as *mut *mut u8,
        );
        let mapped_data = std::slice::from_raw_parts_mut(data_ptr as *mut u8, desc.size as usize);
        crate::CreateBufferMapped {
            id,
            mapped_data,
            detail: CreateBufferMappedDetail { device_id: *device },
        }
    }
}

#[derive(Debug, Hash, PartialEq)]
pub(crate) struct BufferDetail {
    /// On native we need to track the device in order to later destroy the
    /// buffer.
    device_id: DeviceId,
}

pub(crate) fn device_create_buffer_mapped_finish(
    create_buffer_mapped: crate::CreateBufferMapped<'_>,
) -> crate::Buffer {
    buffer_unmap(&create_buffer_mapped.id);
    crate::Buffer {
        id: create_buffer_mapped.id,
        detail: BufferDetail {
            device_id: create_buffer_mapped.detail.device_id,
        },
    }
}

pub(crate) fn buffer_unmap(buffer: &BufferId) {
    wgn::wgpu_buffer_unmap(*buffer);
}

pub(crate) fn device_create_buffer(device: &DeviceId, desc: &BufferDescriptor) -> crate::Buffer {
    let owned_label = OwnedLabel::new(desc.label.as_deref());
    crate::Buffer {
        id: wgn::wgpu_device_create_buffer(
            *device,
            &wgt::BufferDescriptor {
                label: owned_label.as_ptr(),
                size: desc.size,
                usage: desc.usage,
            },
        ),
        detail: BufferDetail { device_id: *device },
    }
}

pub(crate) fn device_create_texture(device: &DeviceId, desc: &TextureDescriptor) -> TextureId {
    let owned_label = OwnedLabel::new(desc.label.as_deref());
    wgn::wgpu_device_create_texture(
        *device,
        &wgt::TextureDescriptor {
            label: owned_label.as_ptr(),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
        },
    )
}

pub(crate) fn device_create_sampler(device: &DeviceId, desc: &SamplerDescriptor) -> SamplerId {
    wgn::wgpu_device_create_sampler(*device, desc)
}

pub(crate) fn create_command_encoder(
    device: &DeviceId,
    desc: &CommandEncoderDescriptor,
) -> CommandEncoderId {
    let owned_label = OwnedLabel::new(desc.label.as_deref());
    wgn::wgpu_device_create_command_encoder(
        *device,
        Some(&wgt::CommandEncoderDescriptor {
            label: owned_label.as_ptr(),
        }),
    )
}

pub(crate) fn command_encoder_copy_buffer_to_buffer(
    command_encoder: &CommandEncoderId,
    source: &crate::Buffer,
    source_offset: wgt::BufferAddress,
    destination: &crate::Buffer,
    destination_offset: wgt::BufferAddress,
    copy_size: wgt::BufferAddress,
) {
    wgn::wgpu_command_encoder_copy_buffer_to_buffer(
        *command_encoder,
        source.id,
        source_offset,
        destination.id,
        destination_offset,
        copy_size,
    );
}

pub(crate) fn command_encoder_copy_buffer_to_texture(
    command_encoder: &CommandEncoderId,
    source: crate::BufferCopyView,
    destination: crate::TextureCopyView,
    copy_size: wgt::Extent3d,
) {
    wgn::wgpu_command_encoder_copy_buffer_to_texture(
        *command_encoder,
        &map_buffer_copy_view(source),
        &map_texture_copy_view(destination),
        copy_size,
    );
}

pub(crate) fn begin_compute_pass(command_encoder: &CommandEncoderId) -> ComputePassId {
    unsafe { wgn::wgpu_command_encoder_begin_compute_pass(*command_encoder, None) }
}

pub(crate) fn compute_pass_set_pipeline(
    compute_pass: &ComputePassId,
    pipeline: &ComputePipelineId,
) {
    unsafe {
        wgn::wgpu_compute_pass_set_pipeline(compute_pass.as_mut().unwrap(), *pipeline);
    }
}

pub(crate) fn compute_pass_set_bind_group<'a>(
    compute_pass: &ComputePassId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[wgt::DynamicOffset],
) {
    unsafe {
        wgn::wgpu_compute_pass_set_bind_group(
            compute_pass.as_mut().unwrap(),
            index,
            *bind_group,
            offsets.as_ptr(),
            offsets.len(),
        );
    }
}

pub(crate) fn compute_pass_dispatch(compute_pass: &ComputePassId, x: u32, y: u32, z: u32) {
    unsafe {
        wgn::wgpu_compute_pass_dispatch(compute_pass.as_mut().unwrap(), x, y, z);
    }
}

pub(crate) fn compute_pass_dispatch_indirect(
    compute_pass: &ComputePassId,
    indirect_buffer: &BufferId,
    indirect_offset: wgt::BufferAddress,
) {
    unsafe {
        wgn::wgpu_compute_pass_dispatch_indirect(
            compute_pass.as_mut().unwrap(),
            *indirect_buffer,
            indirect_offset,
        );
    }
}

pub(crate) fn compute_pass_end_pass(compute_pass: &ComputePassId) {
    unsafe {
        wgn::wgpu_compute_pass_end_pass(*compute_pass);
    }
}

pub(crate) fn command_encoder_finish(command_encoder: &CommandEncoderId) -> CommandBufferId {
    wgn::wgpu_command_encoder_finish(*command_encoder, None)
}

pub(crate) fn queue_submit(queue: &QueueId, command_buffers: &[crate::CommandBuffer]) {
    let temp_command_buffers = command_buffers
        .iter()
        .map(|cb| cb.id)
        .collect::<SmallVec<[_; 4]>>();

    unsafe { wgn::wgpu_queue_submit(*queue, temp_command_buffers.as_ptr(), command_buffers.len()) };
}

pub(crate) fn buffer_map_read(
    buffer: &crate::Buffer,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
) -> impl Future<Output = Result<crate::BufferReadMapping, crate::BufferAsyncErr>> {
    let (future, completion) = native_gpu_future::new_gpu_future(buffer.id, size);

    extern "C" fn buffer_map_read_future_wrapper(
        status: wgc::resource::BufferMapAsyncStatus,
        data: *const u8,
        user_data: *mut u8,
    ) {
        let completion =
            unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
        let (buffer_id, size) = completion.get_buffer_info();

        if let wgc::resource::BufferMapAsyncStatus::Success = status {
            completion.complete(Ok(crate::BufferReadMapping {
                detail: BufferReadMappingDetail {
                    data,
                    size: size as usize,
                    buffer_id,
                },
            }));
        } else {
            completion.complete(Err(crate::BufferAsyncErr));
        }
    }

    wgn::wgpu_buffer_map_read_async(
        buffer.id,
        start,
        size,
        buffer_map_read_future_wrapper,
        completion.to_raw() as _,
    );

    future
}

pub(crate) struct BufferReadMappingDetail {
    data: *const u8,
    size: usize,
    pub(crate) buffer_id: BufferId,
}

impl BufferReadMappingDetail {
    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data as *const u8, self.size) }
    }
}

pub(crate) fn device_create_surface<W: raw_window_handle::HasRawWindowHandle>(
    window: &W,
) -> SurfaceId {
    wgn::wgpu_create_surface(window.raw_window_handle())
}

pub(crate) fn device_create_swap_chain(
    device: &DeviceId,
    surface: &SurfaceId,
    desc: &wgt::SwapChainDescriptor,
) -> SwapChainId {
    wgn::wgpu_device_create_swap_chain(*device, *surface, desc)
}

pub(crate) fn swap_chain_get_next_texture(swap_chain: &SwapChainId) -> Option<TextureViewId> {
    wgn::wgpu_swap_chain_get_next_texture(*swap_chain).view_id
}

pub(crate) fn command_encoder_begin_render_pass<'a>(
    command_encoder: &CommandEncoderId,
    desc: &crate::RenderPassDescriptor<'a, '_>,
) -> RenderPassEncoderId {
    let colors = desc
        .color_attachments
        .iter()
        .map(|ca| wgc::command::RenderPassColorAttachmentDescriptor {
            attachment: ca.attachment.id,
            resolve_target: ca.resolve_target.map(|rt| rt.id),
            load_op: ca.load_op,
            store_op: ca.store_op,
            clear_color: ca.clear_color,
        })
        .collect::<ArrayVec<[_; 4]>>();

    let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
        wgc::command::RenderPassDepthStencilAttachmentDescriptor {
            attachment: dsa.attachment.id,
            depth_load_op: dsa.depth_load_op,
            depth_store_op: dsa.depth_store_op,
            clear_depth: dsa.clear_depth,
            stencil_load_op: dsa.stencil_load_op,
            stencil_store_op: dsa.stencil_store_op,
            clear_stencil: dsa.clear_stencil,
        }
    });

    unsafe {
        wgn::wgpu_command_encoder_begin_render_pass(
            *command_encoder,
            &wgc::command::RenderPassDescriptor {
                color_attachments: colors.as_ptr(),
                color_attachments_length: colors.len(),
                depth_stencil_attachment: depth_stencil.as_ref(),
            },
        )
    }
}

pub(crate) fn render_pass_set_pipeline(
    render_pass: &RenderPassEncoderId,
    pipeline: &RenderPipelineId,
) {
    unsafe {
        wgn::wgpu_render_pass_set_pipeline(render_pass.as_mut().unwrap(), *pipeline);
    }
}

pub(crate) fn render_pass_set_bind_group(
    render_pass: &RenderPassEncoderId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[wgt::DynamicOffset],
) {
    unsafe {
        wgn::wgpu_render_pass_set_bind_group(
            render_pass.as_mut().unwrap(),
            index,
            *bind_group,
            offsets.as_ptr(),
            offsets.len(),
        );
    }
}

pub(crate) fn render_pass_set_index_buffer<'a>(
    render_pass: &RenderPassEncoderId,
    buffer: &'a crate::Buffer,
    offset: wgt::BufferAddress,
    size: wgt::BufferAddress,
) {
    unsafe {
        wgn::wgpu_render_pass_set_index_buffer(
            render_pass.as_mut().unwrap(),
            buffer.id,
            offset,
            size,
        );
    }
}

pub(crate) fn render_pass_set_vertex_buffer<'a>(
    render_pass: &RenderPassEncoderId,
    slot: u32,
    buffer: &'a crate::Buffer,
    offset: wgt::BufferAddress,
    size: wgt::BufferAddress,
) {
    unsafe {
        wgn::wgpu_render_pass_set_vertex_buffer(
            render_pass.as_mut().unwrap(),
            slot,
            buffer.id,
            offset,
            size,
        )
    };
}

pub(crate) fn render_pass_draw(
    render_pass: &RenderPassEncoderId,
    vertices: Range<u32>,
    instances: Range<u32>,
) {
    unsafe {
        wgn::wgpu_render_pass_draw(
            render_pass.as_mut().unwrap(),
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        );
    }
}

pub(crate) fn render_pass_draw_indexed(
    render_pass: &RenderPassEncoderId,
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32>,
) {
    unsafe {
        wgn::wgpu_render_pass_draw_indexed(
            render_pass.as_mut().unwrap(),
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        );
    }
}

pub(crate) fn render_pass_end_pass(render_pass: &RenderPassEncoderId) {
    unsafe {
        wgn::wgpu_render_pass_end_pass(*render_pass);
    }
}

pub(crate) fn texture_create_default_view(texture: &TextureId) -> TextureViewId {
    wgn::wgpu_texture_create_view(*texture, None)
}

pub(crate) fn swap_chain_present(swap_chain: &SwapChainId) {
    wgn::wgpu_swap_chain_present(*swap_chain);
}

pub(crate) fn device_poll(device: &DeviceId, maintain: crate::Maintain) {
    wgn::wgpu_device_poll(
        *device,
        match maintain {
            crate::Maintain::Poll => false,
            crate::Maintain::Wait => true,
        },
    );
}

struct OwnedLabel(Option<CString>);

impl OwnedLabel {
    fn new(text: Option<&str>) -> Self {
        Self(text.map(|t| CString::new(t).expect("invalid label")))
    }

    fn as_ptr(&self) -> *const std::os::raw::c_char {
        match self.0 {
            Some(ref c_string) => c_string.as_ptr(),
            None => ptr::null(),
        }
    }
}
