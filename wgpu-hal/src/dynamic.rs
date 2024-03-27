#[derive(Debug, Clone)]
pub struct Api;
impl crate::Api for Api {
    type Instance = dyn crate::Instance<A = Self>;
    type Surface = dyn crate::Surface<A = Self>;
    type Adapter = dyn crate::Adapter<A = Self>;
    type Device = dyn crate::Device<A = Self>;
    type Queue = dyn crate::Queue<A = Self>;
    type CommandEncoder = dyn crate::CommandEncoder<A = Self>;
    type CommandBuffer = dyn crate::CommandBuffer;

    type Buffer = dyn crate::Buffer;
    type Texture = dyn crate::Texture;
    type SurfaceTexture = dyn crate::SurfaceTexture<dyn crate::Texture>;
    type TextureView = dyn crate::TextureView;
    type Sampler = dyn crate::Sampler;
    type QuerySet = dyn crate::QuerySet;
    type Fence = dyn crate::Fence;

    type BindGroupLayout = dyn crate::BindGroupLayout;
    type BindGroup = dyn crate::BindGroup;
    type PipelineLayout = dyn crate::PipelineLayout;
    type ShaderModule = dyn crate::ShaderModule;
    type RenderPipeline = dyn crate::RenderPipeline;
    type ComputePipeline = dyn crate::ComputePipeline;

    type AccelerationStructure = dyn crate::AccelerationStructure;
}

pub type Instance = dyn crate::Instance<A = Api>;
pub type Surface = dyn crate::Surface<A = Api>;
pub type Adapter = dyn crate::Adapter<A = Api>;
pub type Device = dyn crate::Device<A = Api>;
pub type Queue = dyn crate::Queue<A = Api>;
pub type CommandEncoder = dyn crate::CommandEncoder<A = Api>;
pub type CommandBuffer = dyn crate::CommandBuffer;

pub type Buffer = dyn crate::Buffer;
pub type Texture = dyn crate::Texture;
pub type SurfaceTexture = dyn crate::SurfaceTexture<Texture>;
pub type TextureView = dyn crate::TextureView;
pub type Sampler = dyn crate::Sampler;
pub type QuerySet = dyn crate::QuerySet;
pub type Fence = dyn crate::Fence;

pub type BindGroupLayout = dyn crate::BindGroupLayout;
pub type BindGroup = dyn crate::BindGroup;
pub type PipelineLayout = dyn crate::PipelineLayout;
pub type ShaderModule = dyn crate::ShaderModule;
pub type RenderPipeline = dyn crate::RenderPipeline;
pub type ComputePipeline = dyn crate::ComputePipeline;

pub type AccelerationStructure = dyn crate::AccelerationStructure;

#[derive(Debug)]
pub struct Dyn<T: ?Sized>(pub T);

impl<T: crate::Instance + 'static> crate::Instance for Dyn<T> {
    type A = Api;

    unsafe fn init(_desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError>
    where
        Self: Sized,
    {
        unreachable!()
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        _window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError>
    where
        <Self::A as crate::Api>::Surface: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_surface(&self, _surface: Surface)
    where
        <Self::A as crate::Api>::Surface: Sized,
    {
        unreachable!()
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Self::A>>
    where
        <Self::A as crate::Api>::Adapter: Sized,
    {
        unreachable!()
    }
}

impl<T: crate::Surface> crate::Surface for Dyn<T>
where
    <T::A as crate::Api>::Device: Sized,
{
    type A = Api;

    unsafe fn configure(
        &self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        unsafe {
            self.0
                .configure(device.as_any().downcast_ref().unwrap(), config)
        }
    }

    unsafe fn unconfigure(&self, device: &Device) {
        unsafe { self.0.unconfigure(device.as_any().downcast_ref().unwrap()) }
    }

    unsafe fn acquire_texture(
        &self,
        _timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Self::A>>, crate::SurfaceError>
    where
        <Self::A as crate::Api>::SurfaceTexture: Sized,
    {
        unreachable!()
    }

    unsafe fn discard_texture(&self, _texture: <Self::A as crate::Api>::SurfaceTexture)
    where
        <Self::A as crate::Api>::SurfaceTexture: Sized,
    {
        unreachable!()
    }
}

impl<T: crate::Adapter> crate::Adapter for Dyn<T>
where
    <T::A as crate::Api>::Surface: Sized,
{
    type A = Api;

    unsafe fn open(
        &self,
        _features: wgt::Features,
        _limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<Self::A>, crate::DeviceError>
    where
        <Self::A as crate::Api>::Device: Sized,
        <Self::A as crate::Api>::Queue: Sized,
    {
        unreachable!()
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        unsafe { self.0.texture_format_capabilities(format) }
    }

    unsafe fn surface_capabilities(&self, surface: &Surface) -> Option<crate::SurfaceCapabilities> {
        unsafe {
            self.0
                .surface_capabilities(surface.as_any().downcast_ref().unwrap())
        }
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        unsafe { self.0.get_presentation_timestamp() }
    }
}

impl<T: crate::Device> crate::Device for Dyn<T>
where
    <T::A as crate::Api>::Buffer: Sized,
    <T::A as crate::Api>::Fence: Sized,
{
    type A = Api;

    unsafe fn exit(self, _queue: Queue)
    where
        <Self::A as crate::Api>::Queue: Sized,
    {
        unreachable!()
    }

    unsafe fn create_buffer(
        &self,
        _desc: &crate::BufferDescriptor,
    ) -> Result<Buffer, crate::DeviceError>
    where
        <Self::A as crate::Api>::Buffer: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_buffer(&self, _buffer: Buffer)
    where
        <Self::A as crate::Api>::Buffer: Sized,
    {
        unreachable!()
    }

    unsafe fn map_buffer(
        &self,
        buffer: &Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        unsafe {
            self.0
                .map_buffer(buffer.as_any().downcast_ref().unwrap(), range)
        }
    }

    unsafe fn unmap_buffer(&self, buffer: &Buffer) -> Result<(), crate::DeviceError> {
        unsafe { self.0.unmap_buffer(buffer.as_any().downcast_ref().unwrap()) }
    }

    unsafe fn flush_mapped_ranges(
        &self,
        buffer: &Buffer,
        ranges: &mut dyn Iterator<Item = crate::MemoryRange>,
    ) {
        unsafe {
            self.0
                .flush_mapped_ranges(buffer.as_any().downcast_ref().unwrap(), ranges)
        }
    }

    unsafe fn invalidate_mapped_ranges(
        &self,
        buffer: &Buffer,
        ranges: &mut dyn Iterator<Item = crate::MemoryRange>,
    ) {
        unsafe {
            self.0
                .invalidate_mapped_ranges(buffer.as_any().downcast_ref().unwrap(), ranges)
        }
    }

    unsafe fn create_texture(
        &self,
        _desc: &crate::TextureDescriptor,
    ) -> Result<Texture, crate::DeviceError>
    where
        <Self::A as crate::Api>::Texture: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_texture(&self, _texture: Texture)
    where
        <Self::A as crate::Api>::Texture: Sized,
    {
        unreachable!()
    }

    unsafe fn create_texture_view(
        &self,
        _texture: &Texture,
        _desc: &crate::TextureViewDescriptor,
    ) -> Result<TextureView, crate::DeviceError>
    where
        <Self::A as crate::Api>::TextureView: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_texture_view(&self, _view: TextureView)
    where
        <Self::A as crate::Api>::TextureView: Sized,
    {
        unreachable!()
    }

    unsafe fn create_sampler(
        &self,
        _desc: &crate::SamplerDescriptor,
    ) -> Result<Sampler, crate::DeviceError>
    where
        <Self::A as crate::Api>::Sampler: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_sampler(&self, _sampler: Sampler)
    where
        <Self::A as crate::Api>::Sampler: Sized,
    {
        unreachable!()
    }

    unsafe fn create_command_encoder(
        &self,
        _desc: &crate::CommandEncoderDescriptor<Self::A>,
    ) -> Result<CommandEncoder, crate::DeviceError>
    where
        <Self::A as crate::Api>::CommandEncoder: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_command_encoder(&self, _pool: CommandEncoder)
    where
        <Self::A as crate::Api>::CommandEncoder: Sized,
    {
        unreachable!()
    }

    unsafe fn create_bind_group_layout(
        &self,
        _desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayout, crate::DeviceError>
    where
        <Self::A as crate::Api>::BindGroupLayout: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_bind_group_layout(&self, _bg_layout: BindGroupLayout)
    where
        <Self::A as crate::Api>::BindGroupLayout: Sized,
    {
        unreachable!()
    }

    unsafe fn create_pipeline_layout(
        &self,
        _desc: &crate::PipelineLayoutDescriptor<Self::A>,
    ) -> Result<PipelineLayout, crate::DeviceError>
    where
        <Self::A as crate::Api>::PipelineLayout: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: PipelineLayout)
    where
        <Self::A as crate::Api>::PipelineLayout: Sized,
    {
        unreachable!()
    }

    unsafe fn create_bind_group(
        &self,
        _desc: &crate::BindGroupDescriptor<Self::A>,
    ) -> Result<BindGroup, crate::DeviceError>
    where
        <Self::A as crate::Api>::BindGroup: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_bind_group(&self, _group: BindGroup)
    where
        <Self::A as crate::Api>::BindGroup: Sized,
    {
        unreachable!()
    }

    unsafe fn create_shader_module(
        &self,
        _desc: &crate::ShaderModuleDescriptor,
        _shader: crate::ShaderInput,
    ) -> Result<ShaderModule, crate::ShaderError>
    where
        <Self::A as crate::Api>::ShaderModule: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_shader_module(&self, _module: ShaderModule)
    where
        <Self::A as crate::Api>::ShaderModule: Sized,
    {
        unreachable!()
    }

    unsafe fn create_render_pipeline(
        &self,
        _desc: &crate::RenderPipelineDescriptor<Self::A>,
    ) -> Result<RenderPipeline, crate::PipelineError>
    where
        <Self::A as crate::Api>::RenderPipeline: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_render_pipeline(&self, _pipeline: RenderPipeline)
    where
        <Self::A as crate::Api>::RenderPipeline: Sized,
    {
        unreachable!()
    }

    unsafe fn create_compute_pipeline(
        &self,
        _desc: &crate::ComputePipelineDescriptor<Self::A>,
    ) -> Result<ComputePipeline, crate::PipelineError>
    where
        <Self::A as crate::Api>::ComputePipeline: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_compute_pipeline(&self, _pipeline: ComputePipeline)
    where
        <Self::A as crate::Api>::ComputePipeline: Sized,
    {
        unreachable!()
    }

    unsafe fn create_query_set(
        &self,
        _desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<QuerySet, crate::DeviceError>
    where
        <Self::A as crate::Api>::QuerySet: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_query_set(&self, _set: QuerySet)
    where
        <Self::A as crate::Api>::QuerySet: Sized,
    {
        unreachable!()
    }

    unsafe fn create_fence(&self) -> Result<Fence, crate::DeviceError>
    where
        <Self::A as crate::Api>::Fence: Sized,
    {
        unreachable!()
    }

    unsafe fn destroy_fence(&self, _fence: Fence)
    where
        <Self::A as crate::Api>::Fence: Sized,
    {
        unreachable!()
    }

    unsafe fn get_fence_value(
        &self,
        fence: &Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        unsafe {
            self.0
                .get_fence_value(fence.as_any().downcast_ref().unwrap())
        }
    }

    unsafe fn wait(
        &self,
        fence: &Fence,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        unsafe {
            self.0
                .wait(fence.as_any().downcast_ref().unwrap(), value, timeout_ms)
        }
    }

    unsafe fn start_capture(&self) -> bool {
        unsafe { self.0.start_capture() }
    }

    unsafe fn stop_capture(&self) {
        unsafe { self.0.stop_capture() }
    }

    unsafe fn create_acceleration_structure(
        &self,
        _desc: &crate::AccelerationStructureDescriptor,
    ) -> Result<AccelerationStructure, crate::DeviceError>
    where
        <Self::A as crate::Api>::AccelerationStructure: Sized,
    {
        unreachable!()
    }

    unsafe fn get_acceleration_structure_build_sizes(
        &self,
        _desc: &crate::GetAccelerationStructureBuildSizesDescriptor<Self::A>,
    ) -> crate::AccelerationStructureBuildSizes {
        todo!()
    }

    unsafe fn get_acceleration_structure_device_address(
        &self,
        _acceleration_structure: &AccelerationStructure,
    ) -> wgt::BufferAddress {
        todo!()
    }

    unsafe fn destroy_acceleration_structure(&self, _acceleration_structure: AccelerationStructure)
    where
        <Self::A as crate::Api>::AccelerationStructure: Sized,
    {
        unreachable!()
    }
}

impl<T: crate::Queue> crate::Queue for Dyn<T>
where
    <T::A as crate::Api>::CommandBuffer: Sized,
    <T::A as crate::Api>::SurfaceTexture: Sized,
    <T::A as crate::Api>::Fence: Sized,
{
    type A = Api;

    unsafe fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        surface_textures: &[&SurfaceTexture],
        signal_fence: Option<(&mut Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        let command_buffers = command_buffers
            .iter()
            .map(|buf| buf.as_any().downcast_ref().unwrap())
            .collect::<Vec<_>>();
        let surface_textures = surface_textures
            .iter()
            .map(|buf| buf.as_any().downcast_ref().unwrap())
            .collect::<Vec<_>>();
        unsafe {
            self.0.submit(
                command_buffers.as_slice(),
                surface_textures.as_slice(),
                signal_fence
                    .map(|(fence, value)| (fence.as_any_mut().downcast_mut().unwrap(), value)),
            )
        }
    }

    unsafe fn present(
        &self,
        _surface: &Surface,
        _texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError>
    where
        <Self::A as crate::Api>::SurfaceTexture: Sized,
    {
        unreachable!()
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        unsafe { self.0.get_timestamp_period() }
    }
}

impl<T: crate::CommandEncoder> crate::CommandEncoder for Dyn<T>
where
    <T::A as crate::Api>::Buffer: Sized,
    <T::A as crate::Api>::Texture: Sized,
    <T::A as crate::Api>::PipelineLayout: Sized,
    <T::A as crate::Api>::BindGroup: Sized,
    <T::A as crate::Api>::QuerySet: Sized,
    <T::A as crate::Api>::TextureView: Sized,
    <T::A as crate::Api>::RenderPipeline: Sized,
    <T::A as crate::Api>::ComputePipeline: Sized,
{
    type A = Api;

    unsafe fn begin_encoding(&mut self, label: crate::Label) -> Result<(), crate::DeviceError> {
        unsafe { self.0.begin_encoding(label) }
    }

    unsafe fn discard_encoding(&mut self) {
        unsafe { self.0.discard_encoding() }
    }

    unsafe fn end_encoding(&mut self) -> Result<CommandBuffer, crate::DeviceError>
    where
        <Self::A as crate::Api>::CommandBuffer: Sized,
    {
        unreachable!()
    }

    unsafe fn reset_all(&mut self, _command_buffers: &mut dyn Iterator<Item = CommandBuffer>)
    where
        <Self::A as crate::Api>::CommandBuffer: Sized,
    {
        unreachable!()
    }

    unsafe fn transition_buffers<'a>(
        &mut self,
        barriers: &mut dyn Iterator<Item = crate::BufferBarrier<'a, Self::A>>,
    ) {
        unsafe {
            self.0.transition_buffers(&mut barriers.map(
                |crate::BufferBarrier { buffer, usage }| crate::BufferBarrier {
                    buffer: buffer.as_any().downcast_ref().unwrap(),
                    usage,
                },
            ))
        }
    }

    unsafe fn transition_textures<'a>(
        &mut self,
        barriers: &mut dyn Iterator<Item = crate::TextureBarrier<'a, Self::A>>,
    ) {
        unsafe {
            self.0.transition_textures(&mut barriers.map(
                |crate::TextureBarrier {
                     texture,
                     range,
                     usage,
                 }| crate::TextureBarrier {
                    texture: texture.as_any().downcast_ref().unwrap(),
                    range,
                    usage,
                },
            ))
        }
    }

    unsafe fn clear_buffer(&mut self, buffer: &Buffer, range: crate::MemoryRange) {
        unsafe {
            self.0
                .clear_buffer(buffer.as_any().downcast_ref().unwrap(), range)
        }
    }

    unsafe fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        dst: &Buffer,
        regions: &mut dyn Iterator<Item = crate::BufferCopy>,
    ) {
        unsafe {
            self.0.copy_buffer_to_buffer(
                src.as_any().downcast_ref().unwrap(),
                dst.as_any().downcast_ref().unwrap(),
                regions,
            )
        }
    }

    unsafe fn copy_texture_to_texture(
        &mut self,
        src: &Texture,
        src_usage: crate::TextureUses,
        dst: &Texture,
        regions: &mut dyn Iterator<Item = crate::TextureCopy>,
    ) {
        unsafe {
            self.0.copy_texture_to_texture(
                src.as_any().downcast_ref().unwrap(),
                src_usage,
                dst.as_any().downcast_ref().unwrap(),
                regions,
            )
        }
    }

    unsafe fn copy_buffer_to_texture(
        &mut self,
        src: &Buffer,
        dst: &Texture,
        regions: &mut dyn Iterator<Item = crate::BufferTextureCopy>,
    ) {
        unsafe {
            self.0.copy_buffer_to_texture(
                src.as_any().downcast_ref().unwrap(),
                dst.as_any().downcast_ref().unwrap(),
                regions,
            )
        }
    }

    unsafe fn copy_texture_to_buffer(
        &mut self,
        src: &<Self::A as crate::Api>::Texture,
        src_usage: crate::TextureUses,
        dst: &<Self::A as crate::Api>::Buffer,
        regions: &mut dyn Iterator<Item = crate::BufferTextureCopy>,
    ) {
        unsafe {
            self.0.copy_texture_to_buffer(
                src.as_any().downcast_ref().unwrap(),
                src_usage,
                dst.as_any().downcast_ref().unwrap(),
                regions,
            )
        }
    }

    unsafe fn set_bind_group(
        &mut self,
        layout: &PipelineLayout,
        index: u32,
        group: &BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        unsafe {
            self.0.set_bind_group(
                layout.as_any().downcast_ref().unwrap(),
                index,
                group.as_any().downcast_ref().unwrap(),
                dynamic_offsets,
            )
        }
    }

    unsafe fn set_push_constants(
        &mut self,
        layout: &PipelineLayout,
        stages: wgt::ShaderStages,
        offset_bytes: u32,
        data: &[u32],
    ) {
        unsafe {
            self.0.set_push_constants(
                layout.as_any().downcast_ref().unwrap(),
                stages,
                offset_bytes,
                data,
            )
        }
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {
        unsafe { self.0.insert_debug_marker(label) }
    }

    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        unsafe { self.0.begin_debug_marker(group_label) }
    }

    unsafe fn end_debug_marker(&mut self) {
        unsafe { self.0.end_debug_marker() }
    }

    unsafe fn begin_query(&mut self, set: &QuerySet, index: u32) {
        unsafe {
            self.0
                .begin_query(set.as_any().downcast_ref().unwrap(), index)
        }
    }

    unsafe fn end_query(&mut self, set: &QuerySet, index: u32) {
        unsafe {
            self.0
                .end_query(set.as_any().downcast_ref().unwrap(), index)
        }
    }

    unsafe fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        unsafe {
            self.0
                .write_timestamp(set.as_any().downcast_ref().unwrap(), index)
        }
    }

    unsafe fn reset_queries(
        &mut self,
        set: &<Self::A as crate::Api>::QuerySet,
        range: std::ops::Range<u32>,
    ) {
        unsafe {
            self.0
                .reset_queries(set.as_any().downcast_ref().unwrap(), range)
        }
    }

    unsafe fn copy_query_results(
        &mut self,
        set: &<Self::A as crate::Api>::QuerySet,
        range: std::ops::Range<u32>,
        buffer: &<Self::A as crate::Api>::Buffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
        unsafe {
            self.0.copy_query_results(
                set.as_any().downcast_ref().unwrap(),
                range,
                buffer.as_any().downcast_ref().unwrap(),
                offset,
                stride,
            )
        }
    }

    unsafe fn begin_render_pass(
        &mut self,
        &crate::RenderPassDescriptor {
            label,
            extent,
            sample_count,
            color_attachments,
            ref depth_stencil_attachment,
            multiview,
            ref timestamp_writes,
            occlusion_query_set,
        }: &crate::RenderPassDescriptor<Self::A>,
    ) {
        let color_attachments = color_attachments
            .iter()
            .map(|attachment| {
                attachment.as_ref().map(
                    |&crate::ColorAttachment {
                         ref target,
                         ref resolve_target,
                         ops,
                         clear_value,
                     }| crate::ColorAttachment {
                        target: crate::Attachment {
                            view: target.view.as_any().downcast_ref().unwrap(),
                            usage: target.usage,
                        },
                        resolve_target: resolve_target.as_ref().map(|target| crate::Attachment {
                            view: target.view.as_any().downcast_ref().unwrap(),
                            usage: target.usage,
                        }),
                        ops,
                        clear_value,
                    },
                )
            })
            .collect::<Vec<_>>();
        unsafe {
            self.0.begin_render_pass(&crate::RenderPassDescriptor {
                label,
                extent,
                sample_count,
                color_attachments: color_attachments.as_slice(),
                depth_stencil_attachment: depth_stencil_attachment.as_ref().map(
                    |&crate::DepthStencilAttachment {
                         ref target,
                         depth_ops,
                         stencil_ops,
                         clear_value,
                     }| {
                        crate::DepthStencilAttachment {
                            target: crate::Attachment {
                                view: target.view.as_any().downcast_ref().unwrap(),
                                usage: target.usage,
                            },
                            depth_ops,
                            stencil_ops,
                            clear_value,
                        }
                    },
                ),
                multiview,
                timestamp_writes: timestamp_writes.as_ref().map(
                    |&crate::RenderPassTimestampWrites {
                         query_set,
                         beginning_of_pass_write_index,
                         end_of_pass_write_index,
                     }| {
                        crate::RenderPassTimestampWrites {
                            query_set: query_set.as_any().downcast_ref().unwrap(),
                            beginning_of_pass_write_index,
                            end_of_pass_write_index,
                        }
                    },
                ),
                occlusion_query_set: occlusion_query_set
                    .map(|query_set| query_set.as_any().downcast_ref().unwrap()),
            })
        }
    }

    unsafe fn end_render_pass(&mut self) {
        unsafe { self.0.end_render_pass() }
    }

    unsafe fn set_render_pipeline(&mut self, pipeline: &RenderPipeline) {
        unsafe {
            self.0
                .set_render_pipeline(pipeline.as_any().downcast_ref().unwrap())
        }
    }

    unsafe fn set_index_buffer<'a>(
        &mut self,
        crate::BufferBinding {
            buffer,
            offset,
            size,
        }: crate::BufferBinding<'a, Self::A>,
        format: wgt::IndexFormat,
    ) {
        unsafe {
            self.0.set_index_buffer(
                crate::BufferBinding {
                    buffer: buffer.as_any().downcast_ref().unwrap(),
                    offset,
                    size,
                },
                format,
            )
        }
    }

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        crate::BufferBinding {
            buffer,
            offset,
            size,
        }: crate::BufferBinding<'a, Self::A>,
    ) {
        unsafe {
            self.0.set_vertex_buffer(
                index,
                crate::BufferBinding {
                    buffer: buffer.as_any().downcast_ref().unwrap(),
                    offset,
                    size,
                },
            )
        }
    }

    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: std::ops::Range<f32>) {
        unsafe { self.0.set_viewport(rect, depth_range) }
    }

    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {
        unsafe { self.0.set_scissor_rect(rect) }
    }

    unsafe fn set_stencil_reference(&mut self, value: u32) {
        unsafe { self.0.set_stencil_reference(value) }
    }

    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        unsafe { self.0.set_blend_constants(color) }
    }

    unsafe fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            self.0
                .draw(first_vertex, vertex_count, first_instance, instance_count)
        }
    }

    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            self.0.draw_indexed(
                first_index,
                index_count,
                base_vertex,
                first_instance,
                instance_count,
            )
        }
    }

    unsafe fn draw_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        unsafe {
            self.0
                .draw_indirect(buffer.as_any().downcast_ref().unwrap(), offset, draw_count)
        }
    }

    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        unsafe {
            self.0.draw_indexed_indirect(
                buffer.as_any().downcast_ref().unwrap(),
                offset,
                draw_count,
            )
        }
    }

    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        unsafe {
            self.0.draw_indirect_count(
                buffer.as_any().downcast_ref().unwrap(),
                offset,
                count_buffer.as_any().downcast_ref().unwrap(),
                count_offset,
                max_count,
            )
        }
    }

    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        unsafe {
            self.0.draw_indexed_indirect_count(
                buffer.as_any().downcast_ref().unwrap(),
                offset,
                count_buffer.as_any().downcast_ref().unwrap(),
                count_offset,
                max_count,
            )
        }
    }

    unsafe fn begin_compute_pass(
        &mut self,
        &crate::ComputePassDescriptor {
            label,
            ref timestamp_writes,
        }: &crate::ComputePassDescriptor<Self::A>,
    ) {
        unsafe {
            self.0.begin_compute_pass(&crate::ComputePassDescriptor {
                label,
                timestamp_writes: timestamp_writes.as_ref().map(
                    |&crate::ComputePassTimestampWrites {
                         query_set,
                         beginning_of_pass_write_index,
                         end_of_pass_write_index,
                     }| crate::ComputePassTimestampWrites {
                        query_set: query_set.as_any().downcast_ref().unwrap(),
                        beginning_of_pass_write_index,
                        end_of_pass_write_index,
                    },
                ),
            })
        }
    }

    unsafe fn end_compute_pass(&mut self) {
        unsafe { self.0.end_compute_pass() }
    }

    unsafe fn set_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        unsafe {
            self.0
                .set_compute_pipeline(pipeline.as_any().downcast_ref().unwrap())
        }
    }

    unsafe fn dispatch(&mut self, count: [u32; 3]) {
        unsafe { self.0.dispatch(count) }
    }

    unsafe fn dispatch_indirect(&mut self, buffer: &Buffer, offset: wgt::BufferAddress) {
        unsafe {
            self.0
                .dispatch_indirect(buffer.as_any().downcast_ref().unwrap(), offset)
        }
    }

    unsafe fn build_acceleration_structures<'a>(
        &mut self,
        _descriptor_count: u32,
        _descriptors: &mut dyn Iterator<
            Item = crate::BuildAccelerationStructureDescriptor<'a, Self::A>,
        >,
    ) where
        Self::A: 'a,
    {
        todo!()
    }

    unsafe fn place_acceleration_structure_barrier(
        &mut self,
        barrier: crate::AccelerationStructureBarrier,
    ) {
        unsafe { self.0.place_acceleration_structure_barrier(barrier) }
    }
}
