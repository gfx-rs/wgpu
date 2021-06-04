pub struct Api;
pub struct Surface;
pub struct Adapter;
pub struct Queue;
pub struct Device;
pub struct CommandBuffer;
pub struct RenderPass;
pub struct ComputePass;
pub struct Resource;

impl crate::Surface<Api> for Surface {}

impl crate::Adapter<Api> for Adapter {
    unsafe fn open(
        &self,
        _features: wgt::Features,
    ) -> Result<crate::OpenDevice<Api>, crate::Error> {
        Err(crate::Error::DeviceLost)
    }
    unsafe fn close(&self, _device: Device) {}
    unsafe fn texture_format_capabilities(
        &self,
        _format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapability {
        crate::TextureFormatCapability::empty()
    }
    unsafe fn surface_formats(&self, _surface: &Surface) -> Vec<wgt::TextureFormat> {
        Vec::new()
    }
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit<I: Iterator<Item = CommandBuffer>>(&mut self, _command_buffers: I) {}
}

impl crate::Device<Api> for Device {
    unsafe fn create_buffer(
        &self,
        _desc: &wgt::BufferDescriptor<crate::Label>,
    ) -> Result<Resource, crate::Error> {
        Ok(Resource)
    }
    unsafe fn destroy_buffer(&self, _buffer: Resource) {}
    unsafe fn map_buffer(
        &self,
        _buffer: &Resource,
        _range: crate::MemoryRange,
    ) -> Result<std::ptr::NonNull<u8>, crate::Error> {
        Err(crate::Error::DeviceLost)
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
    ) -> Result<Resource, crate::Error> {
        Ok(Resource)
    }
    unsafe fn destroy_texture(&self, _texture: Resource) {}
    unsafe fn create_texture_view(
        &self,
        _texture: &Resource,
        _desc: &crate::TextureViewDescriptor<crate::Label>,
    ) -> Result<Resource, crate::Error> {
        Ok(Resource)
    }
    unsafe fn destroy_texture_view(&self, _view: Resource) {}
    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor<crate::Label>,
    ) -> Result<Resource, crate::Error> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, _sampler: Resource) {}

    unsafe fn create_command_buffer(&self) -> Result<CommandBuffer, crate::Error> {
        Ok(CommandBuffer)
    }
}

impl crate::CommandBuffer<Api> for CommandBuffer {
    unsafe fn begin(&mut self) {}
    unsafe fn end(&mut self) {}

    unsafe fn begin_render_pass(&mut self) -> RenderPass {
        RenderPass
    }
    unsafe fn end_render_pass(&mut self, _pass: RenderPass) {}
    unsafe fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass
    }
    unsafe fn end_compute_pass(&mut self, _pass: ComputePass) {}
}

impl crate::RenderPass<Api> for RenderPass {}
impl crate::ComputePass<Api> for ComputePass {}

impl crate::Api for Api {
    type Surface = Surface;
    type Adapter = Adapter;
    type Queue = Queue;
    type Device = Device;

    type CommandBuffer = CommandBuffer;
    type RenderPass = RenderPass;
    type ComputePass = ComputePass;

    type Buffer = Resource;
    type QuerySet = Resource;
    type Texture = Resource;
    type SwapChainTexture = Resource;
    type TextureView = Resource;
    type Sampler = Resource;

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        Vec::new()
    }
}
