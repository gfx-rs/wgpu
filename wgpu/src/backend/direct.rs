use crate::{
    backend::native_gpu_future, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource,
    BindingType, BufferDescriptor, CommandEncoderDescriptor, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, TextureDescriptor,
    TextureViewDescriptor, TextureViewDimension,
};

use arrayvec::ArrayVec;
use futures::future::{ready, Ready};
use smallvec::SmallVec;
use std::{ffi::CString, marker::PhantomData, ptr, slice};

macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),+ )) => {
        match $id.backend() {
            #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "vulkan-portability"))]
            wgt::Backend::Vulkan => $global.$method::<wgc::backend::Vulkan>( $($param),+ ),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            wgt::Backend::Metal => $global.$method::<wgc::backend::Metal>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx12 => $global.$method::<wgc::backend::Dx12>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx11 => $global.$method::<wgc::backend::Dx11>( $($param),+ ),
            _ => unreachable!()
        }
    };
}

pub type Context = wgc::hub::Global<wgc::hub::IdentityManagerFactory>;

mod pass_impl {
    use super::Context;
    use std::ops::Range;
    use wgc::command::{compute_ffi::*, render_ffi::*};

    impl crate::ComputePassInner<Context> for wgc::command::RawPass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::ComputePipelineId) {
            unsafe { wgpu_compute_pass_set_pipeline(self, *pipeline) }
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_compute_pass_set_bind_group(
                    self,
                    index,
                    *bind_group,
                    offsets.as_ptr(),
                    offsets.len(),
                )
            }
        }
        fn dispatch(&mut self, x: u32, y: u32, z: u32) {
            unsafe { wgpu_compute_pass_dispatch(self, x, y, z) }
        }
        fn dispatch_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            unsafe { wgpu_compute_pass_dispatch_indirect(self, *indirect_buffer, indirect_offset) }
        }
    }

    impl crate::RenderPassInner<Context> for wgc::command::RawPass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::RenderPipelineId) {
            unsafe { wgpu_render_pass_set_pipeline(self, *pipeline) }
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_render_pass_set_bind_group(
                    self,
                    index,
                    *bind_group,
                    offsets.as_ptr(),
                    offsets.len(),
                )
            }
        }
        fn set_index_buffer(
            &mut self,
            buffer: &wgc::id::BufferId,
            offset: wgt::BufferAddress,
            size: wgt::BufferAddress,
        ) {
            unsafe { wgpu_render_pass_set_index_buffer(self, *buffer, offset, size) }
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &wgc::id::BufferId,
            offset: wgt::BufferAddress,
            size: wgt::BufferAddress,
        ) {
            unsafe { wgpu_render_pass_set_vertex_buffer(self, slot, *buffer, offset, size) }
        }
        fn set_blend_color(&mut self, color: wgt::Color) {
            unsafe { wgpu_render_pass_set_blend_color(self, &color) }
        }
        fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
            unsafe { wgpu_render_pass_set_scissor_rect(self, x, y, width, height) }
        }
        fn set_viewport(
            &mut self,
            x: f32,
            y: f32,
            width: f32,
            height: f32,
            min_depth: f32,
            max_depth: f32,
        ) {
            unsafe {
                wgpu_render_pass_set_viewport(self, x, y, width, height, min_depth, max_depth)
            }
        }
        fn set_stencil_reference(&mut self, reference: u32) {
            unsafe { wgpu_render_pass_set_stencil_reference(self, reference) }
        }
        fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            unsafe {
                wgpu_render_pass_draw(
                    self,
                    vertices.end - vertices.start,
                    instances.end - instances.start,
                    vertices.start,
                    instances.start,
                )
            }
        }
        fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            unsafe {
                wgpu_render_pass_draw_indexed(
                    self,
                    indices.end - indices.start,
                    instances.end - instances.start,
                    indices.start,
                    base_vertex,
                    instances.start,
                )
            }
        }
        fn draw_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            unsafe { wgpu_render_pass_draw_indirect(self, *indirect_buffer, indirect_offset) }
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            unsafe {
                wgpu_render_pass_draw_indexed_indirect(self, *indirect_buffer, indirect_offset)
            }
        }
    }
}

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

impl crate::Context for Context {
    type AdapterId = wgc::id::AdapterId;
    type DeviceId = wgc::id::DeviceId;
    type QueueId = wgc::id::QueueId;
    type ShaderModuleId = wgc::id::ShaderModuleId;
    type BindGroupLayoutId = wgc::id::BindGroupLayoutId;
    type BindGroupId = wgc::id::BindGroupId;
    type TextureViewId = wgc::id::TextureViewId;
    type SamplerId = wgc::id::SamplerId;
    type BufferId = wgc::id::BufferId;
    type TextureId = wgc::id::TextureId;
    type PipelineLayoutId = wgc::id::PipelineLayoutId;
    type RenderPipelineId = wgc::id::RenderPipelineId;
    type ComputePipelineId = wgc::id::ComputePipelineId;
    type CommandEncoderId = wgc::id::CommandEncoderId;
    type ComputePassId = wgc::command::RawPass;
    type CommandBufferId = wgc::id::CommandBufferId;
    type SurfaceId = wgc::id::SurfaceId;
    type SwapChainId = wgc::id::SwapChainId;
    type RenderPassId = wgc::command::RawPass;

    type CreateBufferMappedDetail = CreateBufferMappedDetail;
    type BufferReadMappingDetail = BufferReadMappingDetail;
    type BufferWriteMappingDetail = BufferWriteMappingDetail;
    type SwapChainOutputDetail = SwapChainOutputDetail;

    type RequestAdapterFuture = Ready<Option<Self::AdapterId>>;
    type RequestDeviceFuture =
        Ready<Result<(Self::DeviceId, Self::QueueId), crate::RequestDeviceError>>;
    type MapReadFuture =
        native_gpu_future::GpuFuture<Result<BufferReadMappingDetail, crate::BufferAsyncError>>;
    type MapWriteFuture =
        native_gpu_future::GpuFuture<Result<BufferWriteMappingDetail, crate::BufferAsyncError>>;

    fn init() -> Self {
        wgc::hub::Global::new("wgpu", wgc::hub::IdentityManagerFactory)
    }

    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId {
        self.instance_create_surface(handle, PhantomData)
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_>,
        backends: wgt::BackendBit,
    ) -> Self::RequestAdapterFuture {
        let id = self.pick_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: options.compatible_surface.map(|surface| surface.id),
            },
            wgc::instance::AdapterInputs::Mask(backends, |_| PhantomData),
        );
        ready(id)
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        let device_id = gfx_select!(*adapter => self.adapter_request_device(*adapter, desc, trace_dir, PhantomData));
        ready(Ok((device_id, device_id)))
    }

    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &wgt::SwapChainDescriptor,
    ) -> Self::SwapChainId {
        gfx_select!(*device => self.device_create_swap_chain(*device, *surface, desc))
    }

    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        spv: &[u32],
    ) -> Self::ShaderModuleId {
        let desc = wgc::pipeline::ShaderModuleDescriptor {
            code: wgc::U32Array {
                bytes: spv.as_ptr(),
                length: spv.len(),
            },
        };
        gfx_select!(*device => self.device_create_shader_module(*device, &desc, PhantomData))
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId {
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
        gfx_select!(*device => self.device_create_bind_group_layout(
            *device,
            &bm::BindGroupLayoutDescriptor {
                entries: temp_layouts.as_ptr(),
                entries_length: temp_layouts.len(),
                label: owned_label.as_ptr(),
            },
            PhantomData
        ))
    }

    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId {
        use wgc::binding_model as bm;

        let bindings = desc
            .bindings
            .iter()
            .map(|binding| bm::BindGroupEntry {
                binding: binding.binding,
                resource: match &binding.resource {
                    BindingResource::Buffer(buffer_slice) => {
                        bm::BindingResource::Buffer(bm::BufferBinding {
                            buffer: buffer_slice.buffer.id,
                            offset: buffer_slice.offset,
                            size: buffer_slice.size_or_0(),
                        })
                    }
                    BindingResource::Sampler(ref sampler) => {
                        bm::BindingResource::Sampler(sampler.id)
                    }
                    BindingResource::TextureView(ref texture_view) => {
                        bm::BindingResource::TextureView(texture_view.id)
                    }
                },
            })
            .collect::<Vec<_>>();

        let owned_label = OwnedLabel::new(desc.label.as_deref());
        gfx_select!(*device => self.device_create_bind_group(
            *device,
            &bm::BindGroupDescriptor {
                layout: desc.layout.id,
                entries: bindings.as_ptr(),
                entries_length: bindings.len(),
                label: owned_label.as_ptr(),
            },
            PhantomData
        ))
    }

    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId {
        //TODO: avoid allocation here
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id)
            .collect::<Vec<_>>();

        gfx_select!(*device => self.device_create_pipeline_layout(
            *device,
            &wgc::binding_model::PipelineLayoutDescriptor {
                bind_group_layouts: temp_layouts.as_ptr(),
                bind_group_layouts_length: temp_layouts.len(),
            },
            PhantomData
        ))
    }

    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId {
        use wgc::pipeline as pipe;

        let vertex_entry_point = CString::new(desc.vertex_stage.entry_point).unwrap();
        let vertex_stage = pipe::ProgrammableStageDescriptor {
            module: desc.vertex_stage.module.id,
            entry_point: vertex_entry_point.as_ptr(),
        };
        let (_fragment_entry_point, fragment_stage) =
            if let Some(fragment_stage) = &desc.fragment_stage {
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

        gfx_select!(*device => self.device_create_render_pipeline(
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
            PhantomData
        ))
    }

    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId {
        use wgc::pipeline as pipe;

        let entry_point = CString::new(desc.compute_stage.entry_point).unwrap();

        gfx_select!(*device => self.device_create_compute_pipeline(
            *device,
            &pipe::ComputePipelineDescriptor {
                layout: desc.layout.id,
                compute_stage: pipe::ProgrammableStageDescriptor {
                    module: desc.compute_stage.module.id,
                    entry_point: entry_point.as_ptr(),
                },
            },
            PhantomData
        ))
    }

    fn device_create_buffer_mapped<'a>(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> (Self::BufferId, &'a mut [u8], Self::CreateBufferMappedDetail) {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        unsafe {
            let (id, ptr) = gfx_select!(*device => self.device_create_buffer_mapped(
                *device,
                &desc.map_label(|_| owned_label.as_ptr()),
                PhantomData
            ));
            let mapped_data = std::slice::from_raw_parts_mut(ptr, desc.size as usize);
            (id, mapped_data, CreateBufferMappedDetail)
        }
    }

    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> Self::BufferId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        gfx_select!(*device => self.device_create_buffer(
            *device,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
    }

    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        desc: &TextureDescriptor,
    ) -> Self::TextureId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        gfx_select!(*device => self.device_create_texture(
            *device,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
    }

    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        desc: &SamplerDescriptor,
    ) -> Self::SamplerId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        gfx_select!(*device => self.device_create_sampler(
            *device,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
    }

    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &CommandEncoderDescriptor,
    ) -> Self::CommandEncoderId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        gfx_select!(*device => self.device_create_command_encoder(
            *device,
            &wgt::CommandEncoderDescriptor {
                label: owned_label.as_ptr(),
            },
            PhantomData
        ))
    }

    fn device_drop(&self, device: &Self::DeviceId) {
        #[cfg(not(target_arch = "wasm32"))]
        gfx_select!(*device => self.device_poll(*device, true));
        //TODO: make this work in general
        #[cfg(not(target_arch = "wasm32"))]
        #[cfg(feature = "metal-auto-capture")]
        gfx_select!(*device => self.device_destroy(*device));
    }

    fn device_poll(&self, device: &Self::DeviceId, maintain: crate::Maintain) {
        gfx_select!(*device => self.device_poll(
            *device,
            match maintain {
                crate::Maintain::Poll => false,
                crate::Maintain::Wait => true,
            }
        ));
    }

    fn buffer_map_read(
        &self,
        buffer: &Self::BufferId,
        start: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) -> Self::MapReadFuture {
        let (future, completion) = native_gpu_future::new_gpu_future(*buffer, size);

        extern "C" fn buffer_map_read_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            data: *const u8,
            user_data: *mut u8,
        ) {
            let completion =
                unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
            let (buffer_id, size) = completion.get_buffer_info();

            if let wgc::resource::BufferMapAsyncStatus::Success = status {
                completion.complete(Ok(BufferReadMappingDetail {
                    data,
                    size: size as usize,
                    buffer_id,
                }));
            } else {
                completion.complete(Err(crate::BufferAsyncError));
            }
        }

        let operation = wgc::resource::BufferMapOperation::Read {
            callback: buffer_map_read_future_wrapper,
            userdata: completion.to_raw() as _,
        };
        gfx_select!(*buffer => self.buffer_map_async(*buffer, start .. start + size, operation));

        future
    }

    fn buffer_map_write(
        &self,
        buffer: &Self::BufferId,
        start: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) -> Self::MapWriteFuture {
        let (future, completion) = native_gpu_future::new_gpu_future(*buffer, size);

        extern "C" fn buffer_map_write_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            data: *mut u8,
            user_data: *mut u8,
        ) {
            let completion =
                unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
            let (buffer_id, size) = completion.get_buffer_info();

            if let wgc::resource::BufferMapAsyncStatus::Success = status {
                completion.complete(Ok(BufferWriteMappingDetail {
                    data,
                    size: size as usize,
                    buffer_id,
                }));
            } else {
                completion.complete(Err(crate::BufferAsyncError));
            }
        }

        let operation = wgc::resource::BufferMapOperation::Write {
            callback: buffer_map_write_future_wrapper,
            userdata: completion.to_raw() as _,
        };
        gfx_select!(*buffer => self.buffer_map_async(*buffer, start .. start + size, operation));

        future
    }

    fn buffer_unmap(&self, buffer: &Self::BufferId) {
        gfx_select!(*buffer => self.buffer_unmap(*buffer))
    }

    fn swap_chain_get_next_texture(
        &self,
        swap_chain: &Self::SwapChainId,
    ) -> Result<(Self::TextureViewId, Self::SwapChainOutputDetail), crate::TimeOut> {
        gfx_select!(*swap_chain => self.swap_chain_get_next_texture(*swap_chain, PhantomData))
            .map(|output| {
                (
                    output.view_id.unwrap(),
                    SwapChainOutputDetail {
                        swap_chain_id: *swap_chain,
                    },
                )
            })
            .map_err(|_| crate::TimeOut)
    }

    fn swap_chain_present(&self, view: &Self::TextureViewId, detail: &Self::SwapChainOutputDetail) {
        gfx_select!(*view => self.swap_chain_present(detail.swap_chain_id))
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        desc: Option<&TextureViewDescriptor>,
    ) -> Self::TextureViewId {
        let owned_label = OwnedLabel::new(desc.and_then(|d| d.label.as_deref()));
        let descriptor = desc.map(|d| d.map_label(|_| owned_label.as_ptr()));
        gfx_select!(*texture => self.texture_create_view(*texture, descriptor.as_ref(), PhantomData))
    }

    fn texture_drop(&self, texture: &Self::TextureId) {
        gfx_select!(*texture => self.texture_destroy(*texture))
    }
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId) {
        gfx_select!(*texture_view => self.texture_view_destroy(*texture_view))
    }
    fn sampler_drop(&self, sampler: &Self::SamplerId) {
        gfx_select!(*sampler => self.sampler_destroy(*sampler))
    }
    fn buffer_drop(&self, buffer: &Self::BufferId) {
        gfx_select!(*buffer => self.buffer_destroy(*buffer))
    }
    fn bind_group_drop(&self, bind_group: &Self::BindGroupId) {
        gfx_select!(*bind_group => self.bind_group_destroy(*bind_group))
    }
    fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId) {
        gfx_select!(*bind_group_layout => self.bind_group_layout_destroy(*bind_group_layout))
    }
    fn pipeline_layout_drop(&self, pipeline_layout: &Self::PipelineLayoutId) {
        gfx_select!(*pipeline_layout => self.pipeline_layout_destroy(*pipeline_layout))
    }
    fn shader_module_drop(&self, shader_module: &Self::ShaderModuleId) {
        gfx_select!(*shader_module => self.shader_module_destroy(*shader_module))
    }
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId) {
        gfx_select!(*command_buffer => self.command_buffer_destroy(*command_buffer))
    }
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId) {
        gfx_select!(*pipeline => self.compute_pipeline_destroy(*pipeline))
    }
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId) {
        gfx_select!(*pipeline => self.render_pipeline_destroy(*pipeline))
    }

    fn flush_mapped_data(_data: &mut [u8], _detail: CreateBufferMappedDetail) {}

    fn encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: wgt::BufferAddress,
        destination: &Self::BufferId,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        gfx_select!(*encoder => self.command_encoder_copy_buffer_to_buffer(
            *encoder,
            *source,
            source_offset,
            *destination,
            destination_offset,
            copy_size
        ))
    }

    fn encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::BufferCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        gfx_select!(*encoder => self.command_encoder_copy_buffer_to_texture(
            *encoder,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            copy_size
        ))
    }

    fn encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::BufferCopyView,
        copy_size: wgt::Extent3d,
    ) {
        gfx_select!(*encoder => self.command_encoder_copy_texture_to_buffer(
            *encoder,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            copy_size
        ))
    }

    fn encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        gfx_select!(*encoder => self.command_encoder_copy_texture_to_texture(
            *encoder,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            copy_size
        ))
    }

    fn encoder_begin_compute_pass(&self, encoder: &Self::CommandEncoderId) -> Self::ComputePassId {
        unsafe { wgc::command::RawPass::new_compute(*encoder) }
    }

    fn encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    ) {
        let data = unsafe {
            let mut length = 0;
            let ptr = wgc::command::compute_ffi::wgpu_compute_pass_finish(pass, &mut length);
            slice::from_raw_parts(ptr, length)
        };
        gfx_select!(*encoder => self.command_encoder_run_compute_pass(*encoder, data));
        unsafe { pass.invalidate() };
    }

    fn encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &crate::RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId {
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
            wgc::command::RawPass::new_render(
                *encoder,
                &wgc::command::RenderPassDescriptor {
                    color_attachments: colors.as_ptr(),
                    color_attachments_length: colors.len(),
                    depth_stencil_attachment: depth_stencil.as_ref(),
                },
            )
        }
    }

    fn encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    ) {
        let data = unsafe {
            let mut length = 0;
            let ptr = wgc::command::render_ffi::wgpu_render_pass_finish(pass, &mut length);
            slice::from_raw_parts(ptr, length)
        };
        gfx_select!(*encoder => self.command_encoder_run_render_pass(*encoder, data));
        unsafe { pass.invalidate() };
    }

    fn encoder_finish(&self, encoder: &Self::CommandEncoderId) -> Self::CommandBufferId {
        let desc = wgt::CommandBufferDescriptor::default();
        gfx_select!(*encoder => self.command_encoder_finish(*encoder, &desc))
    }

    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        data: &[u8],
        buffer: &Self::BufferId,
        offset: wgt::BufferAddress,
    ) {
        gfx_select!(*queue => self.queue_write_buffer(*queue, data, *buffer, offset))
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    ) {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();

        gfx_select!(*queue => self.queue_submit(*queue, &temp_command_buffers))
    }
}

pub(crate) struct CreateBufferMappedDetail;

pub(crate) struct BufferReadMappingDetail {
    pub(crate) buffer_id: wgc::id::BufferId,
    data: *const u8,
    size: usize,
}

impl BufferReadMappingDetail {
    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data as *const u8, self.size) }
    }
}

pub(crate) struct BufferWriteMappingDetail {
    pub(crate) buffer_id: wgc::id::BufferId,
    data: *mut u8,
    size: usize,
}

// SAFETY: It is safe to implement Send for `BufferReadMappingDetail` and `BufferWriteMappingDetail`
// because the only !Send field is `data`, and it is used similarly to `&[u8]` or `&mut [u8]`.

unsafe impl Send for BufferReadMappingDetail {}
unsafe impl Send for BufferWriteMappingDetail {}

impl BufferWriteMappingDetail {
    pub(crate) fn as_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data as *mut u8, self.size) }
    }
}

#[derive(Debug)]
pub(crate) struct SwapChainOutputDetail {
    swap_chain_id: wgc::id::SwapChainId,
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
