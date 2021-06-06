#![allow(unused_variables)]

#[derive(Clone)]
pub struct Api;
pub struct Context;
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Api for Api {
    type Instance = Context;
    type Surface = Context;
    type Adapter = Context;
    type Queue = Context;
    type Device = Context;

    type CommandBuffer = Encoder;
    type RenderPass = Encoder;
    type ComputePass = Encoder;

    type Buffer = Resource;
    type QuerySet = Resource;
    type Texture = Resource;
    type SurfaceTexture = Resource;
    type TextureView = Resource;
    type Sampler = Resource;

    type BindGroupLayout = Resource;
    type BindGroup = Resource;
    type PipelineLayout = Resource;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

impl crate::Instance<Api> for Context {
    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        Vec::new()
    }
}

impl crate::Surface<Api> for Context {
    unsafe fn configure(
        &mut self,
        device: &Context,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &Context) {}

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<(Resource, Option<crate::Suboptimal>), crate::SurfaceError> {
        Ok((Resource, None))
    }
}

impl crate::Adapter<Api> for Context {
    unsafe fn open(&self, features: wgt::Features) -> DeviceResult<crate::OpenDevice<Api>> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn close(&self, device: Context) {}
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapability {
        crate::TextureFormatCapability::empty()
    }
    unsafe fn surface_capabilities(&self, surface: &Context) -> Option<crate::SurfaceCapabilities> {
        None
    }
}

impl crate::Queue<Api> for Context {
    unsafe fn submit<I>(&mut self, command_buffers: I) {}
}

impl crate::Device<Api> for Context {
    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_buffer(&self, buffer: Resource) {}
    unsafe fn map_buffer(
        &self,
        buffer: &Resource,
        range: crate::MemoryRange,
    ) -> DeviceResult<std::ptr::NonNull<u8>> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn unmap_buffer(&self, buffer: &Resource) {}
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &Resource, ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &Resource, ranges: I) {}

    unsafe fn create_texture(&self, desc: &crate::TextureDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture(&self, texture: Resource) {}
    unsafe fn create_texture_view(
        &self,
        texture: &Resource,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture_view(&self, view: Resource) {}
    unsafe fn create_sampler(&self, desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, sampler: Resource) {}

    unsafe fn create_command_buffer(
        &self,
        desc: &crate::CommandBufferDescriptor,
    ) -> DeviceResult<Encoder> {
        Ok(Encoder)
    }
    unsafe fn destroy_command_buffer(&self, cmd_buf: Encoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: Resource) {}
    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Resource) {}
    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group(&self, group: Resource) {}

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::NagaShader,
    ) -> Result<Resource, (crate::ShaderError, crate::NagaShader)> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: Resource) {}
}

impl crate::CommandBuffer<Api> for Encoder {
    unsafe fn begin(&mut self) {}
    unsafe fn end(&mut self) {}

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, Api>>,
    {
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, Api>>,
    {
    }

    unsafe fn fill_buffer(&mut self, buffer: &Resource, range: crate::MemoryRange, value: u8) {}

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &Resource, dst: &Resource, regions: T)
    where
        T: Iterator<Item = crate::BufferCopy>,
    {
    }

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &Resource,
        src_usage: crate::TextureUse,
        dst: &Resource,
        regions: T,
    ) {
    }

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Resource, dst: &Resource, regions: T) {}

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Resource,
        src_usage: crate::TextureUse,
        dst: &Resource,
        regions: T,
    ) {
    }

    unsafe fn begin_render_pass(&mut self) -> Encoder {
        Encoder
    }
    unsafe fn end_render_pass(&mut self, pass: Encoder) {}
    unsafe fn begin_compute_pass(&mut self) -> Encoder {
        Encoder
    }
    unsafe fn end_compute_pass(&mut self, pass: Encoder) {}
}

impl crate::RenderPass<Api> for Encoder {
    unsafe fn set_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_bind_group(&mut self, layout: &Resource, index: u32, group: &Resource) {}

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, Api>,
        format: wgt::IndexFormat,
    ) {
    }
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: crate::BufferBinding<'a, Api>) {
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect, depth_range: std::ops::Range<f32>) {}
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect) {}
    unsafe fn set_stencil_reference(&mut self, value: u32) {}
    unsafe fn set_blend_constants(&mut self, color: wgt::Color) {}

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &Resource,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Resource,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
}

impl crate::ComputePass<Api> for Encoder {
    unsafe fn set_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_bind_group(&mut self, layout: &Resource, index: u32, group: &Resource) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_indirect(&mut self, buffer: &Resource, offset: wgt::BufferAddress) {}
}
