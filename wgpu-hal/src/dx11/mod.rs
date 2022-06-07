#![allow(dead_code)]
#![allow(unused_variables)]

use winapi::um::{d3d11, d3d11_1, d3d11_2};

mod adapter;
mod command;
mod device;
mod instance;
mod library;

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

pub struct Instance {
    lib_d3d11: library::D3D11Lib,
    lib_dxgi: native::DxgiLib,
    factory: native::DxgiFactory,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

pub struct Surface {}

pub struct Adapter {
    device: D3D11Device,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

native::weak_com_inheritance_chain! {
    #[derive(Debug, Copy, Clone, PartialEq)]
    enum D3D11Device {
        Device(d3d11::ID3D11Device), from_device, as_device, device;
        Device1(d3d11_1::ID3D11Device1), from_device1, as_device1, unwrap_device1;
        Device2(d3d11_2::ID3D11Device2), from_device2, as_device2, unwrap_device2;
    }
}

pub struct Device {}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

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
        _timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        todo!()
    }

    unsafe fn discard_texture(&mut self, texture: SurfaceTexture) {
        todo!()
    }
}
