use crate::{
    backend::native_gpu_future, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource,
    BufferDescriptor, CommandEncoderDescriptor, ComputePipelineDescriptor, Features, Limits,
    LoadOp, MapMode, Operations, PipelineLayoutDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleSource, SwapChainStatus, TextureDescriptor,
    TextureViewDescriptor,
};

use arrayvec::ArrayVec;
use futures::future::{ready, Ready};
use smallvec::SmallVec;
use std::{borrow::Cow::Borrowed, ffi::CString, fmt, marker::PhantomData, ops::Range, ptr, slice};
use typed_arena::Arena;

pub struct Context(wgc::hub::Global<wgc::hub::IdentityManagerFactory>);

impl Context {
    pub fn adapter_get_info(&self, id: wgc::id::AdapterId) -> wgc::instance::AdapterInfo {
        let global = &self.0;
        wgc::gfx_select!(id => global.adapter_get_info(id))
    }

    pub fn enumerate_adapters(&self, backends: wgt::BackendBit) -> Vec<wgc::id::AdapterId> {
        self.0
            .enumerate_adapters(wgc::instance::AdapterInputs::Mask(backends, |_| {
                PhantomData
            }))
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> crate::Surface {
        let surface = wgc::instance::Surface {
            #[cfg(feature = "vulkan-portability")]
            vulkan: None, //TODO: create_surface_from_layer ?
            metal: self.0.instance.metal.as_ref().map(|inst| {
                inst.create_surface_from_layer(layer as *mut _, cfg!(debug_assertions))
            }),
        };

        crate::Surface {
            id: self.0.surfaces.register_identity(
                PhantomData,
                surface,
                &mut wgc::hub::Token::root(),
            ),
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("type", &"Native").finish()
    }
}

mod pass_impl {
    use super::Context;
    use smallvec::SmallVec;
    use std::convert::TryInto;
    use std::ops::Range;
    use wgc::command::{bundle_ffi::*, compute_ffi::*, render_ffi::*};

    impl crate::ComputePassInner<Context> for wgc::command::ComputePass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::ComputePipelineId) {
            wgpu_compute_pass_set_pipeline(self, *pipeline)
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
        fn set_push_constants(&mut self, offset: u32, data: &[u32]) {
            unsafe {
                wgpu_compute_pass_set_push_constant(
                    self,
                    offset,
                    (data.len() * std::mem::size_of::<u32>())
                        .try_into()
                        .unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn dispatch(&mut self, x: u32, y: u32, z: u32) {
            wgpu_compute_pass_dispatch(self, x, y, z)
        }
        fn dispatch_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_compute_pass_dispatch_indirect(self, *indirect_buffer, indirect_offset)
        }
    }

    impl crate::RenderInner<Context> for wgc::command::RenderPass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::RenderPipelineId) {
            wgpu_render_pass_set_pipeline(self, *pipeline)
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
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_pass_set_index_buffer(self, *buffer, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &wgc::id::BufferId,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_pass_set_vertex_buffer(self, slot, *buffer, offset, size)
        }
        fn set_push_constants(&mut self, stages: wgt::ShaderStage, offset: u32, data: &[u32]) {
            unsafe {
                wgpu_render_pass_set_push_constants(
                    self,
                    stages,
                    offset,
                    (data.len() * std::mem::size_of::<u32>())
                        .try_into()
                        .unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            wgpu_render_pass_draw(
                self,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        }
        fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            wgpu_render_pass_draw_indexed(
                self,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        }
        fn draw_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indirect(self, *indirect_buffer, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indexed_indirect(self, *indirect_buffer, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect(self, *indirect_buffer, indirect_offset, count)
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect(
                self,
                *indirect_buffer,
                indirect_offset,
                count,
            )
        }
        fn multi_draw_indirect_count(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &wgc::id::BufferId,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect_count(
                self,
                *indirect_buffer,
                indirect_offset,
                *count_buffer,
                count_buffer_offset,
                max_count,
            )
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &wgc::id::BufferId,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect_count(
                self,
                *indirect_buffer,
                indirect_offset,
                *count_buffer,
                count_buffer_offset,
                max_count,
            )
        }
    }

    impl crate::RenderPassInner<Context> for wgc::command::RenderPass {
        fn set_blend_color(&mut self, color: wgt::Color) {
            wgpu_render_pass_set_blend_color(self, &color)
        }
        fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
            wgpu_render_pass_set_scissor_rect(self, x, y, width, height)
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
            wgpu_render_pass_set_viewport(self, x, y, width, height, min_depth, max_depth)
        }
        fn set_stencil_reference(&mut self, reference: u32) {
            wgpu_render_pass_set_stencil_reference(self, reference)
        }

        fn insert_debug_marker(&mut self, label: &str) {
            unsafe {
                let label = std::ffi::CString::new(label).unwrap();
                wgpu_render_pass_insert_debug_marker(self, label.as_ptr().into(), 0);
            }
        }

        fn push_debug_group(&mut self, group_label: &str) {
            unsafe {
                let label = std::ffi::CString::new(group_label).unwrap();
                wgpu_render_pass_push_debug_group(self, label.as_ptr().into(), 0);
            }
        }

        fn pop_debug_group(&mut self) {
            wgpu_render_pass_pop_debug_group(self);
        }

        fn execute_bundles<'a, I: Iterator<Item = &'a wgc::id::RenderBundleId>>(
            &mut self,
            render_bundles: I,
        ) {
            wgc::span!(_guard, TRACE, "RenderPass::execute_bundles wrapper");
            let temp_render_bundles = render_bundles.cloned().collect::<SmallVec<[_; 4]>>();
            unsafe {
                wgpu_render_pass_execute_bundles(
                    self,
                    temp_render_bundles.as_ptr(),
                    temp_render_bundles.len(),
                )
            }
        }
    }

    impl crate::RenderInner<Context> for wgc::command::RenderBundleEncoder {
        fn set_pipeline(&mut self, pipeline: &wgc::id::RenderPipelineId) {
            wgpu_render_bundle_set_pipeline(self, *pipeline)
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_render_bundle_set_bind_group(
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
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_bundle_set_index_buffer(self, *buffer, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &wgc::id::BufferId,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_bundle_set_vertex_buffer(self, slot, *buffer, offset, size)
        }

        fn set_push_constants(&mut self, stages: wgt::ShaderStage, offset: u32, data: &[u32]) {
            unsafe {
                wgpu_render_bundle_set_push_constants(
                    self,
                    stages,
                    offset,
                    (data.len() * std::mem::size_of::<u32>())
                        .try_into()
                        .unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            wgpu_render_bundle_draw(
                self,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        }
        fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            wgpu_render_bundle_draw_indexed(
                self,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        }
        fn draw_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_bundle_draw_indirect(self, *indirect_buffer, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &wgc::id::BufferId,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_bundle_indexed_indirect(self, *indirect_buffer, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            _indirect_buffer: &wgc::id::BufferId,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            _indirect_buffer: &wgc::id::BufferId,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indirect_count(
            &mut self,
            _indirect_buffer: &wgc::id::BufferId,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &wgc::id::BufferId,
            _count_buffer_offset: wgt::BufferAddress,
            _max_count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            _indirect_buffer: &wgc::id::BufferId,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &wgc::id::BufferId,
            _count_buffer_offset: wgt::BufferAddress,
            _max_count: u32,
        ) {
            unimplemented!()
        }
    }
}

fn map_buffer_copy_view(view: crate::BufferCopyView) -> wgc::command::BufferCopyView {
    wgc::command::BufferCopyView {
        buffer: view.buffer.id,
        layout: view.layout,
    }
}

fn map_texture_copy_view(view: crate::TextureCopyView) -> wgc::command::TextureCopyView {
    wgc::command::TextureCopyView {
        texture: view.texture.id,
        mip_level: view.mip_level,
        origin: view.origin,
    }
}

fn map_pass_channel<V: Copy + Default>(
    ops: Option<&Operations<V>>,
) -> wgc::command::PassChannel<V> {
    match ops {
        Some(&Operations {
            load: LoadOp::Clear(clear_value),
            store,
        }) => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Clear,
            store_op: if store {
                wgc::command::StoreOp::Store
            } else {
                wgc::command::StoreOp::Clear
            },
            clear_value,
            read_only: false,
        },
        Some(&Operations {
            load: LoadOp::Load,
            store,
        }) => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Load,
            store_op: if store {
                wgc::command::StoreOp::Store
            } else {
                wgc::command::StoreOp::Clear
            },
            clear_value: V::default(),
            read_only: false,
        },
        None => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Load,
            store_op: wgc::command::StoreOp::Store,
            clear_value: V::default(),
            read_only: true,
        },
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
    type ComputePassId = wgc::command::ComputePass;
    type RenderPassId = wgc::command::RenderPass;
    type CommandBufferId = wgc::id::CommandBufferId;
    type RenderBundleEncoderId = wgc::command::RenderBundleEncoder;
    type RenderBundleId = wgc::id::RenderBundleId;
    type SurfaceId = wgc::id::SurfaceId;
    type SwapChainId = wgc::id::SwapChainId;

    type SwapChainOutputDetail = SwapChainOutputDetail;

    type RequestAdapterFuture = Ready<Option<Self::AdapterId>>;
    type RequestDeviceFuture =
        Ready<Result<(Self::DeviceId, Self::QueueId), crate::RequestDeviceError>>;
    type MapAsyncFuture = native_gpu_future::GpuFuture<Result<(), crate::BufferAsyncError>>;

    fn init(backends: wgt::BackendBit) -> Self {
        Self(wgc::hub::Global::new(
            "wgpu",
            wgc::hub::IdentityManagerFactory,
            backends,
        ))
    }

    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId {
        self.0.instance_create_surface(handle, PhantomData)
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions,
    ) -> Self::RequestAdapterFuture {
        let id = self.0.pick_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: options.compatible_surface.map(|surface| surface.id),
            },
            wgc::instance::AdapterInputs::Mask(wgt::BackendBit::all(), |_| PhantomData),
        );
        ready(id)
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        let global = &self.0;
        let device_id = wgc::gfx_select!(*adapter => global.adapter_request_device(*adapter, desc, trace_dir, PhantomData)).unwrap();
        ready(Ok((device_id, device_id)))
    }

    fn adapter_features(&self, adapter: &Self::AdapterId) -> Features {
        let global = &self.0;
        wgc::gfx_select!(*adapter => global.adapter_features(*adapter))
    }

    fn adapter_limits(&self, adapter: &Self::AdapterId) -> Limits {
        let global = &self.0;
        wgc::gfx_select!(*adapter => global.adapter_limits(*adapter))
    }

    fn device_features(&self, device: &Self::DeviceId) -> Features {
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_features(*device))
    }

    fn device_limits(&self, device: &Self::DeviceId) -> Limits {
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_limits(*device))
    }

    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &wgt::SwapChainDescriptor,
    ) -> Self::SwapChainId {
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_swap_chain(*device, *surface, desc))
    }

    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        source: ShaderModuleSource,
    ) -> Self::ShaderModuleId {
        let desc = match source {
            ShaderModuleSource::SpirV(spv) => wgc::pipeline::ShaderModuleSource::SpirV(spv),
            ShaderModuleSource::Wgsl(code) => wgc::pipeline::ShaderModuleSource::Wgsl(code),
        };
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_shader_module(*device, desc, PhantomData))
            .unwrap()
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId {
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_bind_group_layout(
            *device,
            desc,
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId {
        wgc::span!(_guard, TRACE, "Device::create_bind_group wrapper");
        use wgc::binding_model as bm;

        let texture_view_arena: Arena<wgc::id::TextureViewId> = Arena::new();
        let entries = desc
            .entries
            .iter()
            .map(|entry| bm::BindGroupEntry {
                binding: entry.binding,
                resource: match entry.resource {
                    BindingResource::Buffer(ref buffer_slice) => {
                        bm::BindingResource::Buffer(bm::BufferBinding {
                            buffer_id: buffer_slice.buffer.id,
                            offset: buffer_slice.offset,
                            size: buffer_slice.size,
                        })
                    }
                    BindingResource::Sampler(ref sampler) => {
                        bm::BindingResource::Sampler(sampler.id)
                    }
                    BindingResource::TextureView(ref texture_view) => {
                        bm::BindingResource::TextureView(texture_view.id)
                    }
                    BindingResource::TextureViewArray(texture_view_array) => {
                        bm::BindingResource::TextureViewArray(Borrowed(
                            texture_view_arena
                                .alloc_extend(texture_view_array.iter().map(|view| view.id)),
                        ))
                    }
                },
            })
            .collect::<Vec<_>>();

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_bind_group(
            *device,
            &bm::BindGroupDescriptor {
                label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
                layout: desc.layout.id,
                entries: Borrowed(&entries),
            },
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId {
        wgc::span!(_guard, TRACE, "Device::create_pipeline_layout wrapper");

        // Limit is always less or equal to wgc::MAX_BIND_GROUPS, so this is always right
        // Guards following ArrayVec
        assert!(
            desc.bind_group_layouts.len() <= wgc::MAX_BIND_GROUPS,
            "Bind group layout count {} exceeds device bind group limit {}",
            desc.bind_group_layouts.len(),
            wgc::MAX_BIND_GROUPS
        );

        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id)
            .collect::<ArrayVec<[_; wgc::MAX_BIND_GROUPS]>>();

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_pipeline_layout(
            *device,
            &wgt::PipelineLayoutDescriptor {
                bind_group_layouts: Borrowed(&temp_layouts),
                push_constant_ranges: Borrowed(&desc.push_constant_ranges),
            },
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId {
        wgc::span!(_guard, TRACE, "Device::create_render_pipeline wrapper");
        use wgc::pipeline as pipe;

        let vertex_stage = pipe::ProgrammableStageDescriptor {
            module: desc.vertex_stage.module.id,
            entry_point: Borrowed(&desc.vertex_stage.entry_point),
        };
        let fragment_stage =
            desc.fragment_stage
                .as_ref()
                .map(|fs| pipe::ProgrammableStageDescriptor {
                    module: fs.module.id,
                    entry_point: Borrowed(&fs.entry_point),
                });
        let vertex_state = wgt::VertexStateDescriptor {
            index_format: desc.vertex_state.index_format,
            vertex_buffers: Borrowed(&desc.vertex_state.vertex_buffers),
        };

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_render_pipeline(
            *device,
            &pipe::RenderPipelineDescriptor {
                layout: desc.layout.id,
                vertex_stage,
                fragment_stage,
                rasterization_state: desc.rasterization_state.clone(),
                primitive_topology: desc.primitive_topology,
                color_states: Borrowed(&desc.color_states),
                depth_stencil_state: desc.depth_stencil_state.clone(),
                vertex_state: vertex_state,
                sample_count: desc.sample_count,
                sample_mask: desc.sample_mask,
                alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
            },
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId {
        use wgc::pipeline as pipe;

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_compute_pipeline(
            *device,
            &pipe::ComputePipelineDescriptor {
                layout: desc.layout.id,
                compute_stage: pipe::ProgrammableStageDescriptor {
                    module: desc.compute_stage.module.id,
                    entry_point: Borrowed(&desc.compute_stage.entry_point),
                },
            },
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> Self::BufferId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_buffer(
            *device,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        desc: &TextureDescriptor,
    ) -> Self::TextureId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_texture(
            *device,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
        .unwrap()
    }

    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        desc: &SamplerDescriptor,
    ) -> Self::SamplerId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_sampler(
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

        let global = &self.0;
        wgc::gfx_select!(*device => global.device_create_command_encoder(
            *device,
            &wgt::CommandEncoderDescriptor {
                label: owned_label.as_ptr(),
            },
            PhantomData
        ))
    }

    fn device_create_render_bundle_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &wgt::RenderBundleEncoderDescriptor,
    ) -> Self::RenderBundleEncoderId {
        wgc::command::RenderBundleEncoder::new(desc, *device, None).unwrap()
    }

    fn device_drop(&self, device: &Self::DeviceId) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let global = &self.0;
            wgc::gfx_select!(*device => global.device_poll(*device, true)).unwrap();
        }
        //TODO: make this work in general
        #[cfg(not(target_arch = "wasm32"))]
        #[cfg(feature = "metal-auto-capture")]
        {
            let global = &self.0;
            wgc::gfx_select!(*device => global.device_destroy(*device));
        }
    }

    fn device_poll(&self, device: &Self::DeviceId, maintain: crate::Maintain) {
        let global = &self.0;
        wgc::gfx_select!(*device => global.device_poll(
            *device,
            match maintain {
                crate::Maintain::Poll => false,
                crate::Maintain::Wait => true,
            }
        ))
        .unwrap();
    }

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        mode: MapMode,
        range: Range<wgt::BufferAddress>,
    ) -> Self::MapAsyncFuture {
        wgc::span!(_guard, TRACE, "Buffer::buffer_map_async wrapper");

        let (future, completion) = native_gpu_future::new_gpu_future();

        extern "C" fn buffer_map_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            user_data: *mut u8,
        ) {
            let completion =
                unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
            completion.complete(match status {
                wgc::resource::BufferMapAsyncStatus::Success => Ok(()),
                _ => Err(crate::BufferAsyncError),
            })
        }

        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: buffer_map_future_wrapper,
            user_data: completion.to_raw() as _,
        };

        let global = &self.0;
        wgc::gfx_select!(*buffer => global.buffer_map_async(*buffer, range, operation)).unwrap();

        future
    }

    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<wgt::BufferAddress>,
    ) -> &[u8] {
        let size = sub_range.end - sub_range.start;
        let global = &self.0;
        let ptr = wgc::gfx_select!(*buffer => global.buffer_get_mapped_range(
            *buffer,
            sub_range.start,
            wgt::BufferSize::new(size)
        ))
        .unwrap();
        unsafe { slice::from_raw_parts(ptr, size as usize) }
    }

    fn buffer_get_mapped_range_mut(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<wgt::BufferAddress>,
    ) -> &mut [u8] {
        let size = sub_range.end - sub_range.start;
        let global = &self.0;
        let ptr = wgc::gfx_select!(*buffer => global.buffer_get_mapped_range(
            *buffer,
            sub_range.start,
            wgt::BufferSize::new(size)
        ))
        .unwrap();
        unsafe { slice::from_raw_parts_mut(ptr, size as usize) }
    }

    fn buffer_unmap(&self, buffer: &Self::BufferId) {
        let global = &self.0;
        wgc::gfx_select!(*buffer => global.buffer_unmap(*buffer)).unwrap();
    }

    fn swap_chain_get_current_texture_view(
        &self,
        swap_chain: &Self::SwapChainId,
    ) -> (
        Option<Self::TextureViewId>,
        SwapChainStatus,
        Self::SwapChainOutputDetail,
    ) {
        let global = &self.0;
        let wgc::swap_chain::SwapChainOutput { status, view_id } =
            wgc::gfx_select!(*swap_chain => global.swap_chain_get_current_texture_view(*swap_chain, PhantomData)).unwrap();

        (
            view_id,
            status,
            SwapChainOutputDetail {
                swap_chain_id: *swap_chain,
            },
        )
    }

    fn swap_chain_present(&self, view: &Self::TextureViewId, detail: &Self::SwapChainOutputDetail) {
        let global = &self.0;
        wgc::gfx_select!(*view => global.swap_chain_present(detail.swap_chain_id)).unwrap();
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        desc: Option<&TextureViewDescriptor>,
    ) -> Self::TextureViewId {
        let owned_label = OwnedLabel::new(desc.and_then(|d| d.label.as_deref()));
        let descriptor = desc.map(|d| d.map_label(|_| owned_label.as_ptr()));
        let global = &self.0;
        wgc::gfx_select!(*texture => global.texture_create_view(*texture, descriptor.as_ref(), PhantomData))
    }

    fn texture_drop(&self, texture: &Self::TextureId) {
        let global = &self.0;
        wgc::gfx_select!(*texture => global.texture_destroy(*texture))
    }
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId) {
        let global = &self.0;
        wgc::gfx_select!(*texture_view => global.texture_view_destroy(*texture_view))
    }
    fn sampler_drop(&self, sampler: &Self::SamplerId) {
        let global = &self.0;
        wgc::gfx_select!(*sampler => global.sampler_destroy(*sampler))
    }
    fn buffer_drop(&self, buffer: &Self::BufferId) {
        let global = &self.0;
        wgc::gfx_select!(*buffer => global.buffer_destroy(*buffer))
    }
    fn bind_group_drop(&self, bind_group: &Self::BindGroupId) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group => global.bind_group_destroy(*bind_group))
    }
    fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group_layout => global.bind_group_layout_destroy(*bind_group_layout))
    }
    fn pipeline_layout_drop(&self, pipeline_layout: &Self::PipelineLayoutId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline_layout => global.pipeline_layout_destroy(*pipeline_layout))
    }
    fn shader_module_drop(&self, shader_module: &Self::ShaderModuleId) {
        let global = &self.0;
        wgc::gfx_select!(*shader_module => global.shader_module_destroy(*shader_module))
    }
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId) {
        let global = &self.0;
        wgc::gfx_select!(*command_buffer => global.command_buffer_destroy(*command_buffer))
    }
    fn render_bundle_drop(&self, render_bundle: &Self::RenderBundleId) {
        let global = &self.0;
        wgc::gfx_select!(*render_bundle => global.render_bundle_destroy(*render_bundle))
    }
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.compute_pipeline_destroy(*pipeline))
    }
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.render_pipeline_destroy(*pipeline))
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: wgt::BufferAddress,
        destination: &Self::BufferId,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_copy_buffer_to_buffer(
            *encoder,
            *source,
            source_offset,
            *destination,
            destination_offset,
            copy_size
        ))
        .unwrap()
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::BufferCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_copy_buffer_to_texture(
            *encoder,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        ))
        .unwrap()
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::BufferCopyView,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_copy_texture_to_buffer(
            *encoder,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size
        ))
        .unwrap()
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_copy_texture_to_texture(
            *encoder,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        ))
        .unwrap()
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
    ) -> Self::ComputePassId {
        wgc::command::ComputePass::new(*encoder)
    }

    fn command_encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_run_compute_pass(*encoder, pass))
            .unwrap()
    }

    fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &crate::RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId {
        wgc::span!(_guard, TRACE, "CommandEncoder::begin_render_pass wrapper");
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| wgc::command::ColorAttachmentDescriptor {
                attachment: ca.attachment.id,
                resolve_target: ca.resolve_target.map(|rt| rt.id),
                channel: map_pass_channel(Some(&ca.ops)),
            })
            .collect::<ArrayVec<[_; wgc::device::MAX_COLOR_TARGETS]>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::DepthStencilAttachmentDescriptor {
                attachment: dsa.attachment.id,
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        wgc::command::RenderPass::new(
            *encoder,
            wgc::command::RenderPassDescriptor {
                color_attachments: Borrowed(&colors),
                depth_stencil_attachment: depth_stencil.as_ref(),
            },
        )
    }

    fn command_encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_run_render_pass(*encoder, pass))
            .unwrap()
    }

    fn command_encoder_finish(&self, encoder: &Self::CommandEncoderId) -> Self::CommandBufferId {
        let desc = wgt::CommandBufferDescriptor::default();
        let global = &self.0;
        wgc::gfx_select!(*encoder => global.command_encoder_finish(*encoder, &desc)).unwrap()
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder: Self::RenderBundleEncoderId,
        desc: &crate::RenderBundleDescriptor,
    ) -> Self::RenderBundleId {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        let global = &self.0;
        wgc::gfx_select!(encoder.parent() => global.render_bundle_encoder_finish(
            encoder,
            &desc.map_label(|_| owned_label.as_ptr()),
            PhantomData
        ))
        .unwrap()
    }

    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        buffer: &Self::BufferId,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        let global = &self.0;
        wgc::gfx_select!(*queue => global.queue_write_buffer(*queue, *buffer, offset, data))
    }

    fn queue_write_texture(
        &self,
        queue: &Self::QueueId,
        texture: crate::TextureCopyView,
        data: &[u8],
        data_layout: wgt::TextureDataLayout,
        size: wgt::Extent3d,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*queue => global.queue_write_texture(
            *queue,
            &map_texture_copy_view(texture),
            data,
            &data_layout,
            &size
        ))
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    ) {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();

        let global = &self.0;
        wgc::gfx_select!(*queue => global.queue_submit(*queue, &temp_command_buffers)).unwrap()
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
