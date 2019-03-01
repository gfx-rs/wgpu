extern crate arrayvec;
extern crate wgpu_native as wgn;

use arrayvec::ArrayVec;

use std::ffi::CString;
use std::ops::Range;
use std::ptr;
use std::slice;

pub use wgn::winit;
pub use wgn::{
    AdapterDescriptor, BindGroupLayoutBinding, BindingType,
    BlendDescriptor, BlendOperation, BlendFactor, BufferMapAsyncStatus, ColorWriteFlags,
    RasterizationStateDescriptor, CullMode, FrontFace,
    BufferDescriptor, BufferUsageFlags,
    IndexFormat, InputStepMode, ShaderAttributeIndex, VertexAttributeDescriptor, VertexFormat,
    Color, CommandEncoderDescriptor,
    ColorStateDescriptor, DepthStencilStateDescriptor, StencilStateFaceDescriptor, StencilOperation,
    DeviceDescriptor, Extensions, Extent3d, LoadOp, Origin3d, PowerPreference, PrimitiveTopology,
    RenderPassColorAttachmentDescriptor, RenderPassDepthStencilAttachmentDescriptor,
    ShaderModuleDescriptor, ShaderStageFlags, StoreOp, SwapChainDescriptor,
    SamplerDescriptor, AddressMode, FilterMode, BorderColor, CompareFunction,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsageFlags,
    TextureViewDescriptor, TextureViewDimension, TextureAspectFlags,
};


//TODO: avoid heap allocating vectors during resource creation.
#[derive(Default)]
struct Temp {
    //bind_group_descriptors: Vec<wgn::BindGroupDescriptor>,
    //vertex_buffers: Vec<wgn::VertexBufferDescriptor>,
    command_buffers: Vec<wgn::CommandBufferId>,
}


pub struct Instance {
    id: wgn::InstanceId,
}

pub struct Adapter {
    id: wgn::AdapterId,
}

pub struct Device {
    id: wgn::DeviceId,
    temp: Temp,
}

pub struct Buffer {
    id: wgn::BufferId,
}

pub struct Texture {
    id: wgn::TextureId,
    owned: bool,
}

pub struct TextureView {
    id: wgn::TextureViewId,
    owned: bool,
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

pub struct BindGroupLayout {
    id: wgn::BindGroupLayoutId,
}

pub struct BindGroup {
    id: wgn::BindGroupId,
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        wgn::wgpu_bind_group_destroy(self.id);
    }
}

pub struct ShaderModule {
    id: wgn::ShaderModuleId,
}

pub struct PipelineLayout {
    id: wgn::PipelineLayoutId,
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

pub struct CommandEncoder {
    id: wgn::CommandEncoderId,
}

pub struct RenderPass<'a> {
    id: wgn::RenderPassId,
    _parent: &'a mut CommandEncoder,
}

pub struct ComputePass<'a> {
    id: wgn::ComputePassId,
    _parent: &'a mut CommandEncoder,
}

pub struct Queue<'a> {
    id: wgn::QueueId,
    temp: &'a mut Temp,
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
    pub entry_point: &'a str,
}

#[derive(Clone, Debug)]
pub struct VertexBufferDescriptor<'a> {
    pub stride: u32,
    pub step_mode: InputStepMode,
    pub attributes: &'a [VertexAttributeDescriptor],
}

pub struct RenderPipelineDescriptor<'a> {
    pub layout: &'a PipelineLayout,
    pub vertex_stage: PipelineStageDescriptor<'a>,
    pub fragment_stage: PipelineStageDescriptor<'a>,
    pub rasterization_state: RasterizationStateDescriptor,
    pub primitive_topology: PrimitiveTopology,
    pub color_states: &'a [ColorStateDescriptor],
    pub depth_stencil_state: Option<DepthStencilStateDescriptor>,
    pub index_format: IndexFormat,
    pub vertex_buffers: &'a [VertexBufferDescriptor<'a>],
    pub sample_count: u32,
}

pub struct ComputePipelineDescriptor<'a> {
    pub layout: &'a PipelineLayout,
    pub compute_stage: PipelineStageDescriptor<'a>,
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

    pub fn create_surface(&self, window: &winit::Window) -> Surface {
        Surface {
            id: wgn::wgpu_instance_create_surface_from_winit(self.id, window),
        }
    }

    #[cfg(feature = "metal")]
    pub fn create_surface_with_metal_layer(&self, window: *mut std::ffi::c_void) -> Surface {
        Surface {
            id: wgn::wgpu_instance_create_surface_from_macos_layer(self.id, window),
        }
    }
}

impl Adapter {
    pub fn create_device(&self, desc: &DeviceDescriptor) -> Device {
        Device {
            id: wgn::wgpu_adapter_create_device(self.id, desc),
            temp: Temp::default(),
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
            temp: &mut self.temp,
        }
    }

    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        CommandEncoder {
            id: wgn::wgpu_device_create_command_encoder(self.id, desc),
        }
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        let bindings = desc
            .bindings
            .into_iter()
            .map(|binding| wgn::Binding {
                binding: binding.binding,
                resource: match binding.resource {
                    BindingResource::Buffer {
                        ref buffer,
                        ref range,
                    } => wgn::BindingResource::Buffer(wgn::BufferBinding {
                        buffer: buffer.id,
                        offset: range.start,
                        size: range.end,
                    }),
                    BindingResource::Sampler(ref sampler) => {
                        wgn::BindingResource::Sampler(sampler.id)
                    }
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

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        let vertex_entry_point = CString::new(desc.vertex_stage.entry_point).unwrap();
        let fragment_entry_point = CString::new(desc.fragment_stage.entry_point).unwrap();

        let temp_color_states = desc.color_states.to_vec();
        let temp_vertex_buffers = desc
            .vertex_buffers
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
                    vertex_stage: wgn::PipelineStageDescriptor {
                        module: desc.vertex_stage.module.id,
                        entry_point: vertex_entry_point.as_ptr(),
                    },
                    fragment_stage: wgn::PipelineStageDescriptor {
                        module: desc.fragment_stage.module.id,
                        entry_point: fragment_entry_point.as_ptr(),
                    },
                    rasterization_state: desc.rasterization_state.clone(),
                    primitive_topology: desc.primitive_topology,
                    color_states: temp_color_states.as_ptr(),
                    color_states_length: temp_color_states.len(),
                    depth_stencil_state: desc.depth_stencil_state
                        .as_ref()
                        .map_or(ptr::null(), |p| p as *const _),
                    vertex_buffer_state: wgn::VertexBufferStateDescriptor {
                        index_format: desc.index_format,
                        vertex_buffers: temp_vertex_buffers.as_ptr(),
                        vertex_buffers_count: temp_vertex_buffers.len(),
                    },
                    sample_count: desc.sample_count,
                },
            ),
        }
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        let entry_point = CString::new(desc.compute_stage.entry_point).unwrap();

        ComputePipeline {
            id: wgn::wgpu_device_create_compute_pipeline(
                self.id,
                &wgn::ComputePipelineDescriptor {
                    layout: desc.layout.id,
                    compute_stage: wgn::PipelineStageDescriptor {
                        module: desc.compute_stage.module.id,
                        entry_point: entry_point.as_ptr(),
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

    pub fn create_buffer_mapped<T>(&self, desc: &BufferDescriptor) -> (Buffer, &mut [T])
            where T: 'static + Copy {
        let type_size = std::mem::size_of::<T>() as u32;
        assert_ne!(type_size, 0);
        assert_eq!(desc.size % type_size, 0);

        let mut ptr : *mut u8 = std::ptr::null_mut();

        let buffer = Buffer {
            id: wgn::wgpu_device_create_buffer_mapped(self.id, desc, &mut ptr as *mut *mut u8),
        };

        let data = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, desc.size as usize / std::mem::size_of::<T>()) };

        (buffer, data)
    }

    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture {
        Texture {
            id: wgn::wgpu_device_create_texture(self.id, desc),
            owned: true,
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

impl Drop for Device {
    fn drop(&mut self) {
        //TODO: make this work in general
        #[cfg(feature = "metal-auto-capture")]
        wgn::wgpu_device_destroy(self.id);
    }
}

pub enum BufferMapAsyncResult<T> {
    Success(T),
    Error,
}

struct BufferMapReadAsyncUserData<T,F> 
    where F: FnOnce(BufferMapAsyncResult<&[T]>) {
    size: u32,
    callback: F,
    phantom: std::marker::PhantomData<T>,
}

struct BufferMapWriteAsyncUserData<T, F>
    where F: FnOnce(BufferMapAsyncResult<&mut [T]>) {
    size: u32,
    callback: F,
    phantom: std::marker::PhantomData<T>,
}

impl Buffer {
    pub fn set_sub_data(&self, offset: u32, data: &[u8]) {
        wgn::wgpu_buffer_set_sub_data(self.id, offset, data.len() as u32, data.as_ptr());
    }

    pub fn map_read_async<T, F>(&self, start: u32, size: u32, callback: F)
            where T: 'static + Copy, F: FnOnce(BufferMapAsyncResult<&[T]>) {
        let type_size = std::mem::size_of::<T>() as u32;
        assert_ne!(type_size, 0);
        assert_eq!(size % type_size, 0);

        extern "C" fn buffer_map_read_callback_wrapper<T, F>(status: wgn::BufferMapAsyncStatus, data: *const u8, userdata: *mut u8)
                where F: FnOnce(BufferMapAsyncResult<&[T]>) {
            let userdata = unsafe { Box::from_raw(userdata as *mut BufferMapReadAsyncUserData<T, F>) };
            let data = unsafe { slice::from_raw_parts(data as *const T, userdata.size as usize / std::mem::size_of::<T>()) };
            if let wgn::BufferMapAsyncStatus::Success = status {
                (userdata.callback)(BufferMapAsyncResult::Success::<&[T]>(data));
            } else {
                (userdata.callback)(BufferMapAsyncResult::Error);
            }
        }

        let userdata = Box::new(BufferMapReadAsyncUserData{size, callback, phantom: std::marker::PhantomData});
        wgn::wgpu_buffer_map_read_async(self.id, start, size, buffer_map_read_callback_wrapper::<T, F>, Box::into_raw(userdata) as *mut u8);
    }

    pub fn map_write_async<T, F>(&self, start: u32, size: u32, callback: F)
            where T: 'static + Copy, F: FnOnce(BufferMapAsyncResult<&mut [T]>) {
        let type_size = std::mem::size_of::<T>() as u32;
        assert_ne!(type_size, 0);
        assert_eq!(size % type_size, 0);

        extern "C" fn buffer_map_write_callback_wrapper<T, F>(status: wgn::BufferMapAsyncStatus, data: *mut u8, userdata: *mut u8)
                where F: FnOnce(BufferMapAsyncResult<&mut [T]>) {
            let userdata = unsafe { Box::from_raw(userdata as *mut BufferMapWriteAsyncUserData<T, F>) };
            let data = unsafe { slice::from_raw_parts_mut(data as *mut T, userdata.size as usize / std::mem::size_of::<T>()) };
            if let wgn::BufferMapAsyncStatus::Success = status {
                (userdata.callback)(BufferMapAsyncResult::Success::<&mut [T]>(data));
            } else {
                (userdata.callback)(BufferMapAsyncResult::Error);
            }
        }

        let userdata = Box::new(BufferMapWriteAsyncUserData{size, callback, phantom: std::marker::PhantomData});
        wgn::wgpu_buffer_map_write_async(self.id, start, size, buffer_map_write_callback_wrapper::<T, F>, Box::into_raw(userdata) as *mut u8);
    }

    pub fn unmap(&self) {
        wgn::wgpu_buffer_unmap(self.id);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        wgn::wgpu_buffer_destroy(self.id);
    }
}

impl Texture {
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_view(self.id, desc),
            owned: true,
        }
    }

    pub fn create_default_view(&self) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_default_view(self.id),
            owned: true,
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if self.owned {
            wgn::wgpu_texture_destroy(self.id);
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if self.owned {
            wgn::wgpu_texture_view_destroy(self.id);
        }
    }
}

impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            id: wgn::wgpu_command_encoder_finish(self.id),
        }
    }

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
            id: wgn::wgpu_command_encoder_begin_render_pass(
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
            _parent: self,
        }
    }

    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: wgn::wgpu_command_encoder_begin_compute_pass(self.id),
            _parent: self,
        }
    }

    pub fn copy_buffer_to_buffer(
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

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        wgn::wgpu_render_pass_end_pass(self.id);
    }
}

impl<'a> ComputePass<'a> {
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

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        wgn::wgpu_compute_pass_end_pass(self.id);
    }
}

impl<'a> Queue<'a> {
    pub fn submit(&mut self, command_buffers: &[CommandBuffer]) {
        self.temp.command_buffers.clear();
        self.temp.command_buffers.extend(
            command_buffers.iter().map(|cb| cb.id)
        );

        wgn::wgpu_queue_submit(
            self.id,
            self.temp.command_buffers.as_ptr(),
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
            texture: Texture {
                id: output.texture_id,
                owned: false,
            },
            view: TextureView {
                id: output.view_id,
                owned: false,
            },
            swap_chain_id: &self.id,
        }
    }
}
