extern crate arrayvec;
extern crate wgpu_native as wgn;

use arrayvec::ArrayVec;

use std::ffi::CString;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr;

pub use wgn::{
    AdapterDescriptor, Attachment, BindGroupLayoutBinding, BindingType, BlendStateDescriptor,
    BufferDescriptor, BufferUsageFlags,
    IndexFormat, VertexFormat, InputStepMode, ShaderAttributeIndex, VertexAttributeDescriptor,
    Color, ColorWriteFlags, CommandBufferDescriptor, DepthStencilStateDescriptor,
    DeviceDescriptor, Extensions, Extent3d, LoadOp, Origin3d, PowerPreference, PrimitiveTopology,
    RenderPassColorAttachmentDescriptor, RenderPassDepthStencilAttachmentDescriptor,
    ShaderModuleDescriptor, ShaderStage, ShaderStageFlags, StoreOp, SwapChainDescriptor,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsageFlags, TextureViewDescriptor,
    SamplerDescriptor, AddressMode, FilterMode, CompareFunction, BorderColor,
};

pub struct Instance {
    id: wgn::InstanceId,
}

pub struct Adapter {
    id: wgn::AdapterId,
}

pub struct Device {
    id: wgn::DeviceId,
}

pub struct Buffer {
    id: wgn::BufferId,
}

pub struct Texture {
    id: wgn::TextureId,
}

pub struct TextureView {
    id: wgn::TextureViewId,
}

pub struct Sampler {
    id: wgn::SamplerId,
}

pub struct Surface {
    id: wgn::SurfaceId,
}

pub struct SwapChain {
    id: wgn::SwapChainId,
}

pub enum BindingResource<'a> {
    Buffer {
        buffer: &'a Buffer,
        range: Range<u32>,
    },
    Sampler(&'a Sampler),
    TextureView(&'a TextureView),
}

pub struct Binding<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

pub struct BindGroupLayout {
    id: wgn::BindGroupLayoutId,
}

pub struct BindGroup {
    id: wgn::BindGroupId,
}

pub struct ShaderModule {
    id: wgn::ShaderModuleId,
}

pub struct PipelineLayout {
    id: wgn::PipelineLayoutId,
}

pub struct BlendState {
    id: wgn::BlendStateId,
}

pub struct DepthStencilState {
    id: wgn::DepthStencilStateId,
}

pub struct RenderPipeline {
    id: wgn::RenderPipelineId,
}

pub struct ComputePipeline {
    id: wgn::ComputePipelineId,
}

pub struct CommandBuffer {
    id: wgn::CommandBufferId,
}

pub struct RenderPass<'a> {
    id: wgn::RenderPassId,
    parent: &'a mut CommandBuffer,
}

pub struct ComputePass<'a> {
    id: wgn::ComputePassId,
    parent: &'a mut CommandBuffer,
}

pub struct Queue<'a> {
    id: wgn::QueueId,
    _marker: PhantomData<&'a Self>,
}

pub struct BindGroupLayoutDescriptor<'a> {
    pub bindings: &'a [BindGroupLayoutBinding],
}

pub struct BindGroupDescriptor<'a> {
    pub layout: &'a BindGroupLayout,
    pub bindings: &'a [Binding<'a>],
}

pub struct PipelineLayoutDescriptor<'a> {
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

pub struct PipelineStageDescriptor<'a> {
    pub module: &'a ShaderModule,
    pub stage: ShaderStage,
    pub entry_point: &'a str,
}

pub struct AttachmentsState<'a> {
    pub color_attachments: &'a [Attachment],
    pub depth_stencil_attachment: Option<Attachment>,
}

pub struct VertexBufferDescriptor<'a> {
    pub stride: u32,
    pub step_mode: InputStepMode,
    pub attributes: &'a [VertexAttributeDescriptor],
}

pub struct RenderPipelineDescriptor<'a> {
    pub layout: &'a PipelineLayout,
    pub stages: &'a [PipelineStageDescriptor<'a>],
    pub primitive_topology: PrimitiveTopology,
    pub attachments_state: AttachmentsState<'a>,
    pub blend_states: &'a [&'a BlendState],
    pub depth_stencil_state: &'a DepthStencilState,
    pub index_format: IndexFormat,
    pub vertex_buffers: &'a [VertexBufferDescriptor<'a>],
}

pub struct RenderPassDescriptor<'a> {
    pub color_attachments: &'a [RenderPassColorAttachmentDescriptor<&'a TextureView>],
    pub depth_stencil_attachment:
        Option<RenderPassDepthStencilAttachmentDescriptor<&'a TextureView>>,
}

pub struct SwapChainOutput<'a> {
    pub texture: Texture,
    pub view: TextureView,
    swap_chain_id: &'a wgn::SwapChainId,
}

pub struct BufferCopyView<'a> {
    pub buffer: &'a Buffer,
    pub offset: u32,
    pub row_pitch: u32,
    pub image_height: u32,
}

impl<'a> BufferCopyView<'a> {
    fn into_native(self) -> wgn::BufferCopyView {
        wgn::BufferCopyView {
            buffer: self.buffer.id,
            offset: self.offset,
            row_pitch: self.row_pitch,
            image_height: self.image_height,
        }
    }
}

pub struct TextureCopyView<'a> {
    pub texture: &'a Texture,
    pub level: u32,
    pub slice: u32,
    pub origin: Origin3d,
}

impl<'a> TextureCopyView<'a> {
    fn into_native(self) -> wgn::TextureCopyView {
        wgn::TextureCopyView {
            texture: self.texture.id,
            level: self.level,
            slice: self.slice,
            origin: self.origin,
        }
    }
}


impl Instance {
    pub fn new() -> Self {
        Instance {
            id: wgn::wgpu_create_instance(),
        }
    }

    pub fn get_adapter(&self, desc: &AdapterDescriptor) -> Adapter {
        Adapter {
            id: wgn::wgpu_instance_get_adapter(self.id, desc),
        }
    }

    #[cfg(feature = "winit")]
    pub fn create_surface(&self, window: &wgn::winit::Window) -> Surface {
        Surface {
            id: wgn::wgpu_instance_create_surface_from_winit(self.id, window),
        }
    }
}

impl Adapter {
    pub fn create_device(&self, desc: &DeviceDescriptor) -> Device {
        Device {
            id: wgn::wgpu_adapter_create_device(self.id, desc),
        }
    }
}

impl Device {
    pub fn create_shader_module(&self, spv: &[u8]) -> ShaderModule {
        let desc = wgn::ShaderModuleDescriptor {
            code: wgn::ByteArray {
                bytes: spv.as_ptr(),
                length: spv.len(),
            },
        };
        ShaderModule {
            id: wgn::wgpu_device_create_shader_module(self.id, &desc),
        }
    }

    pub fn get_queue(&mut self) -> Queue {
        Queue {
            id: wgn::wgpu_device_get_queue(self.id),
            _marker: PhantomData,
        }
    }

    pub fn create_command_buffer(&self, desc: &CommandBufferDescriptor) -> CommandBuffer {
        CommandBuffer {
            id: wgn::wgpu_device_create_command_buffer(self.id, desc),
        }
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        let bindings = desc
            .bindings
            .into_iter()
            .map(|binding| wgn::Binding {
                binding: binding.binding,
                resource: match binding.resource {
                    BindingResource::Buffer { ref buffer, ref range } => {
                        wgn::BindingResource::Buffer(wgn::BufferBinding {
                            buffer: buffer.id,
                            offset: range.start,
                            size: range.end,
                        })
                    }
                    BindingResource::Sampler(ref sampler) => wgn::BindingResource::Sampler(sampler.id),
                    BindingResource::TextureView(ref texture_view) => {
                        wgn::BindingResource::TextureView(texture_view.id)
                    }
                },
            })
            .collect::<Vec<_>>();
        BindGroup {
            id: wgn::wgpu_device_create_bind_group(
                self.id,
                &wgn::BindGroupDescriptor {
                    layout: desc.layout.id,
                    bindings: bindings.as_ptr(),
                    bindings_length: bindings.len(),
                },
            ),
        }
    }

    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        BindGroupLayout {
            id: wgn::wgpu_device_create_bind_group_layout(
                self.id,
                &wgn::BindGroupLayoutDescriptor {
                    bindings: desc.bindings.as_ptr(),
                    bindings_length: desc.bindings.len(),
                },
            ),
        }
    }

    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        //TODO: avoid allocation here
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id)
            .collect::<Vec<_>>();
        PipelineLayout {
            id: wgn::wgpu_device_create_pipeline_layout(
                self.id,
                &wgn::PipelineLayoutDescriptor {
                    bind_group_layouts: temp_layouts.as_ptr(),
                    bind_group_layouts_length: temp_layouts.len(),
                },
            ),
        }
    }

    pub fn create_blend_state(&self, desc: &BlendStateDescriptor) -> BlendState {
        BlendState {
            id: wgn::wgpu_device_create_blend_state(self.id, desc),
        }
    }

    pub fn create_depth_stencil_state(
        &self,
        desc: &DepthStencilStateDescriptor,
    ) -> DepthStencilState {
        DepthStencilState {
            id: wgn::wgpu_device_create_depth_stencil_state(self.id, desc),
        }
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        let entry_points = desc
            .stages
            .iter()
            .map(|ps| CString::new(ps.entry_point).unwrap())
            .collect::<ArrayVec<[_; 2]>>();
        let stages = desc
            .stages
            .iter()
            .zip(&entry_points)
            .map(|(ps, ep_name)| wgn::PipelineStageDescriptor {
                module: ps.module.id,
                stage: ps.stage,
                entry_point: ep_name.as_ptr(),
            })
            .collect::<ArrayVec<[_; 2]>>();

        let temp_blend_states = desc.blend_states.iter().map(|bs| bs.id).collect::<Vec<_>>();
        let temp_vertex_buffers = desc.vertex_buffers
            .iter()
            .map(|vbuf| wgn::VertexBufferDescriptor {
                stride: vbuf.stride,
                step_mode: vbuf.step_mode,
                attributes: vbuf.attributes.as_ptr(),
                attributes_count: vbuf.attributes.len(),
            })
            .collect::<Vec<_>>();

        RenderPipeline {
            id: wgn::wgpu_device_create_render_pipeline(
                self.id,
                &wgn::RenderPipelineDescriptor {
                    layout: desc.layout.id,
                    stages: stages.as_ptr(),
                    stages_length: stages.len(),
                    primitive_topology: desc.primitive_topology,
                    attachments_state: wgn::AttachmentsState {
                        color_attachments: desc.attachments_state.color_attachments.as_ptr(),
                        color_attachments_length: desc.attachments_state.color_attachments.len(),
                        depth_stencil_attachment: desc
                            .attachments_state
                            .depth_stencil_attachment
                            .as_ref()
                            .map(|at| at as *const _)
                            .unwrap_or(ptr::null()),
                    },
                    blend_states: temp_blend_states.as_ptr(),
                    blend_states_length: temp_blend_states.len(),
                    depth_stencil_state: desc.depth_stencil_state.id,
                    vertex_buffer_state: wgn::VertexBufferStateDescriptor {
                        index_format: desc.index_format,
                        vertex_buffers: temp_vertex_buffers.as_ptr(),
                        vertex_buffers_count: temp_vertex_buffers.len(),
                    },
                },
            ),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        Buffer {
            id: wgn::wgpu_device_create_buffer(self.id, desc),
        }
    }

    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture {
        Texture {
            id: wgn::wgpu_device_create_texture(self.id, desc),
        }
    }

    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler {
        Sampler {
            id: wgn::wgpu_device_create_sampler(self.id, desc),
        }
    }

    pub fn create_swap_chain(&self, surface: &Surface, desc: &SwapChainDescriptor) -> SwapChain {
        SwapChain {
            id: wgn::wgpu_device_create_swap_chain(self.id, surface.id, desc),
        }
    }
}

impl Buffer {
    pub fn set_sub_data(&self, offset: u32, data: &[u8]) {
        wgn::wgpu_buffer_set_sub_data(self.id, offset, data.len() as u32, data.as_ptr());
    }
}

impl Texture {
    pub fn create_texture_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_texture_view(self.id, desc),
        }
    }

    pub fn create_default_texture_view(&self) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_default_texture_view(self.id),
        }
    }
}

impl CommandBuffer {
    pub fn begin_render_pass(&mut self, desc: &RenderPassDescriptor) -> RenderPass {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| RenderPassColorAttachmentDescriptor {
                attachment: ca.attachment.id,
                load_op: ca.load_op,
                store_op: ca.store_op,
                clear_color: ca.clear_color,
            })
            .collect::<ArrayVec<[_; 4]>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            RenderPassDepthStencilAttachmentDescriptor {
                attachment: dsa.attachment.id,
                depth_load_op: dsa.depth_load_op,
                depth_store_op: dsa.depth_store_op,
                clear_depth: dsa.clear_depth,
                stencil_load_op: dsa.stencil_load_op,
                stencil_store_op: dsa.stencil_store_op,
                clear_stencil: dsa.clear_stencil,
            }
        });

        RenderPass {
            id: wgn::wgpu_command_buffer_begin_render_pass(
                self.id,
                wgn::RenderPassDescriptor {
                    color_attachments: colors.as_ptr(),
                    color_attachments_length: colors.len(),
                    depth_stencil_attachment: depth_stencil
                        .as_ref()
                        .map(|at| at as *const _)
                        .unwrap_or(ptr::null()),
                },
            ),
            parent: self,
        }
    }

    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: wgn::wgpu_command_buffer_begin_compute_pass(self.id),
            parent: self,
        }
    }

    pub fn copy_buffer_tobuffer(
        &mut self,
        source: &Buffer,
        source_offset: u32,
        destination: &Buffer,
        destination_offset: u32,
        copy_size: u32,
    ) {
        wgn::wgpu_command_buffer_copy_buffer_to_buffer(
            self.id,
            source.id,
            source_offset,
            destination.id,
            destination_offset,
            copy_size,
        );
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        source: BufferCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        wgn::wgpu_command_buffer_copy_buffer_to_texture(
            self.id,
            &source.into_native(),
            &destination.into_native(),
            copy_size,
        );
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    ) {
        wgn::wgpu_command_buffer_copy_texture_to_buffer(
            self.id,
            &source.into_native(),
            &destination.into_native(),
            copy_size,
        );
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: TextureCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        wgn::wgpu_command_buffer_copy_texture_to_texture(
            self.id,
            &source.into_native(),
            &destination.into_native(),
            copy_size,
        );
    }
}

impl<'a> RenderPass<'a> {
    pub fn end_pass(self) -> &'a mut CommandBuffer {
        wgn::wgpu_render_pass_end_pass(self.id);
        self.parent
    }

    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup) {
        wgn::wgpu_render_pass_set_bind_group(self.id, index, bind_group.id);
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        wgn::wgpu_render_pass_set_pipeline(self.id, pipeline.id);
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u32) {
        wgn::wgpu_render_pass_set_index_buffer(self.id, buffer.id, offset);
    }

    pub fn set_vertex_buffers(&mut self, buffer_pairs: &[(&Buffer, u32)]) {
        let mut buffers = Vec::new();
        let mut offsets = Vec::new();
        for &(buffer, offset) in buffer_pairs {
            buffers.push(buffer.id);
            offsets.push(offset);
        }
        wgn::wgpu_render_pass_set_vertex_buffers(
            self.id,
            buffers.as_ptr(),
            offsets.as_ptr(),
            buffer_pairs.len(),
        );
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        wgn::wgpu_render_pass_draw(
            self.id,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        );
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        wgn::wgpu_render_pass_draw_indexed(
            self.id,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        );
    }
}

impl<'a> ComputePass<'a> {
    pub fn end_pass(self) -> &'a mut CommandBuffer {
        wgn::wgpu_compute_pass_end_pass(self.id);
        self.parent
    }

    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup) {
        wgn::wgpu_compute_pass_set_bind_group(self.id, index, bind_group.id);
    }

    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        wgn::wgpu_compute_pass_set_pipeline(self.id, pipeline.id);
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        wgn::wgpu_compute_pass_dispatch(self.id, x, y, z);
    }
}

impl<'a> Queue<'a> {
    pub fn submit(&mut self, command_buffers: &[CommandBuffer]) {
        wgn::wgpu_queue_submit(
            self.id,
            command_buffers.as_ptr() as *const _,
            command_buffers.len(),
        );
    }
}

impl<'a> Drop for SwapChainOutput<'a> {
    fn drop(&mut self) {
        wgn::wgpu_swap_chain_present(*self.swap_chain_id);
    }
}

impl SwapChain {
    pub fn get_next_texture(&mut self) -> SwapChainOutput {
        let output = wgn::wgpu_swap_chain_get_next_texture(self.id);
        SwapChainOutput {
            texture: Texture { id: output.texture_id },
            view: TextureView { id: output.view_id },
            swap_chain_id: &self.id,
        }
    }
}
