mod adapter;
mod command;
mod device;
mod instance;

#[derive(Clone)]
pub struct Api;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = CommandEncoder;
    type CommandBuffer = CommandBuffer;

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = SurfaceTexture;
    type TextureView = TextureView;
    type Sampler = Sampler;
    type QuerySet = QuerySet;
    type Fence = Fence;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = ShaderModule;
    type RenderPipeline = RenderPipeline;
    type ComputePipeline = ComputePipeline;
}

pub struct Instance {}

pub struct Surface {}

pub struct Adapter {}

pub struct Device {}

pub struct Queue {}

pub struct CommandEncoder {}

pub struct CommandBuffer {}

#[derive(Debug)]
pub struct Buffer {}
#[derive(Debug)]
pub struct Texture {}
#[derive(Debug)]
pub struct SurfaceTexture {}

impl std::borrow::Borrow<Texture> for SurfaceTexture {
    fn borrow(&self) -> &Texture {
        todo!()
    }
}

#[derive(Debug)]
pub struct TextureView {}
#[derive(Debug)]
pub struct Sampler {}
#[derive(Debug)]
pub struct QuerySet {}
#[derive(Debug)]
pub struct Fence {}
#[derive(Debug)]

pub struct BindGroupLayout {}
#[derive(Debug)]
pub struct BindGroup {}
#[derive(Debug)]
pub struct PipelineLayout {}
#[derive(Debug)]
pub struct ShaderModule {}
pub struct RenderPipeline {}
pub struct ComputePipeline {}

impl crate::Surface<Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        todo!()
    }

    unsafe fn unconfigure(&mut self, device: &Device) {
        todo!()
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        todo!()
    }

    unsafe fn discard_texture(&mut self, texture: SurfaceTexture) {
        todo!()
    }
}
