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
        _device: &Context,
        _config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }

    unsafe fn unconfigure(&mut self, _device: &Context) {}

    unsafe fn acquire_texture(
        &mut self,
        _timeout_ms: u32,
    ) -> Result<(Resource, Option<crate::Suboptimal>), crate::SurfaceError> {
        Ok((Resource, None))
    }
}

impl crate::Adapter<Api> for Context {
    unsafe fn open(&self, _features: wgt::Features) -> DeviceResult<crate::OpenDevice<Api>> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn close(&self, _device: Context) {}
    unsafe fn texture_format_capabilities(
        &self,
        _format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapability {
        crate::TextureFormatCapability::empty()
    }
    unsafe fn surface_capabilities(
        &self,
        _surface: &Context,
    ) -> Option<crate::SurfaceCapabilities> {
        None
    }
}

impl crate::Queue<Api> for Context {
    unsafe fn submit<I: Iterator<Item = Encoder>>(&mut self, _command_buffers: I) {}
}

impl crate::Device<Api> for Context {
    unsafe fn create_buffer(
        &self,
        _desc: &wgt::BufferDescriptor<crate::Label>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_buffer(&self, _buffer: Resource) {}
    unsafe fn map_buffer(
        &self,
        _buffer: &Resource,
        _range: crate::MemoryRange,
    ) -> DeviceResult<std::ptr::NonNull<u8>> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn unmap_buffer(&self, _buffer: &Resource) {}
    unsafe fn flush_mapped_ranges<I: Iterator<Item = crate::MemoryRange>>(
        &self,
        _buffer: &Resource,
        _ranges: I,
    ) {
    }
    unsafe fn invalidate_mapped_ranges<I: Iterator<Item = crate::MemoryRange>>(
        &self,
        _buffer: &Resource,
        _ranges: I,
    ) {
    }

    unsafe fn create_texture(
        &self,
        _desc: &wgt::TextureDescriptor<crate::Label>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture(&self, _texture: Resource) {}
    unsafe fn create_texture_view(
        &self,
        _texture: &Resource,
        _desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture_view(&self, _view: Resource) {}
    unsafe fn create_sampler(&self, _desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, _sampler: Resource) {}

    unsafe fn create_command_buffer(
        &self,
        _desc: &crate::CommandBufferDescriptor,
    ) -> DeviceResult<Encoder> {
        Ok(Encoder)
    }
    unsafe fn destroy_command_buffer(&self, _cmd_buf: Encoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        _desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group_layout(&self, _bg_layout: Resource) {}
    unsafe fn create_pipeline_layout(
        &self,
        _desc: &crate::PipelineLayoutDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: Resource) {}
    unsafe fn create_bind_group(
        &self,
        _desc: &crate::BindGroupDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group(&self, _group: Resource) {}

    unsafe fn create_shader_module(
        &self,
        _desc: &crate::ShaderModuleDescriptor,
        _shader: crate::NagaShader,
    ) -> Result<Resource, (crate::ShaderError, crate::NagaShader)> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, _module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        _desc: &crate::RenderPipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, _pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        _desc: &crate::ComputePipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, _pipeline: Resource) {}
}

impl crate::CommandBuffer<Api> for Encoder {
    unsafe fn begin(&mut self) {}
    unsafe fn end(&mut self) {}

    unsafe fn begin_render_pass(&mut self) -> Encoder {
        Encoder
    }
    unsafe fn end_render_pass(&mut self, _pass: Encoder) {}
    unsafe fn begin_compute_pass(&mut self) -> Encoder {
        Encoder
    }
    unsafe fn end_compute_pass(&mut self, _pass: Encoder) {}
}

impl crate::RenderPass<Api> for Encoder {}
impl crate::ComputePass<Api> for Encoder {}
