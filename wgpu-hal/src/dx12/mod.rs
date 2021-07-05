/*!
# DirectX12 API internals.

## Pipeline Layout

!*/

#![allow(unused_variables)]

mod adapter;
mod conv;

use std::{borrow::Cow, ops::Range, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_4, dxgi1_6, windef, winerror},
    Interface as _,
};

#[derive(Clone)]
pub struct Api;
//TODO: remove these temporaries
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = Encoder;
    type CommandBuffer = Resource;

    type Buffer = Resource;
    type Texture = Resource;
    type SurfaceTexture = Resource;
    type TextureView = Resource;
    type Sampler = Resource;
    type QuerySet = Resource;
    type Fence = Resource;

    type BindGroupLayout = Resource;
    type BindGroup = Resource;
    type PipelineLayout = Resource;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

trait HResult {
    fn to_error(self) -> Option<Cow<'static, str>>;
}
impl HResult for i32 {
    fn to_error(self) -> Option<Cow<'static, str>> {
        if self >= 0 {
            return None;
        }
        let description = match self {
            winerror::E_UNEXPECTED => "unexpected",
            winerror::E_NOTIMPL => "not implemented",
            winerror::E_OUTOFMEMORY => "out of memory",
            winerror::E_INVALIDARG => "invalid argument",
            _ => return Some(Cow::Owned(format!("0x{:X}", self as u32))),
        };
        Some(Cow::Borrowed(description))
    }
}

trait HResultPair {
    type Object;
    fn check(self) -> Result<Self::Object, Cow<'static, str>>;
}
impl<T> HResultPair for (T, i32) {
    type Object = T;
    fn check(self) -> Result<T, Cow<'static, str>> {
        match self.1.to_error() {
            None => Ok(self.0),
            Some(err) => Err(err),
        }
    }
}

pub struct Instance {
    factory: native::Factory4,
    library: Arc<native::D3D12Lib>,
    lib_dxgi: native::DxgiLib,
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.factory.destroy();
        }
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

pub struct Surface {
    factory: native::WeakPtr<dxgi1_4::IDXGIFactory4>,
    wnd_handle: windef::HWND,
    //presentation: Option<Presentation>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug, Clone, Copy)]
enum MemoryArchitecture {
    Unified { cache_coherent: bool },
    NonUnified,
}

#[derive(Debug, Clone, Copy)]
struct PrivateCapabilities {
    heterogeneous_resource_heaps: bool,
    memory_architecture: MemoryArchitecture,
}

#[derive(Default)]
struct Workarounds {
    // On WARP, temporary CPU descriptors are still used by the runtime
    // after we call `CopyDescriptors`.
    avoid_cpu_descriptor_overwrites: bool,
}

pub struct Adapter {
    raw: native::WeakPtr<dxgi1_2::IDXGIAdapter2>,
    device: native::Device,
    library: Arc<native::D3D12Lib>,
    private_caps: PrivateCapabilities,
    workarounds: Workarounds,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

pub struct Device {
    raw: native::Device,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: native::CommandQueue,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl crate::Instance<Api> for Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let lib_main = native::D3D12Lib::new().map_err(|_| crate::InstanceError)?;

        let lib_dxgi = native::DxgiLib::new().map_err(|_| crate::InstanceError)?;
        let mut factory_flags = native::FactoryCreationFlags::empty();

        if desc.flags.contains(crate::InstanceFlags::VALIDATION) {
            // Enable debug layer
            match lib_main.get_debug_interface() {
                Ok(pair) => match pair.check() {
                    Ok(debug_controller) => {
                        debug_controller.enable_layer();
                        debug_controller.Release();
                    }
                    Err(err) => {
                        log::warn!("Unable to enable D3D12 debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for D3D12 not found: {:?}", err);
                }
            }

            // The `DXGI_CREATE_FACTORY_DEBUG` flag is only allowed to be passed to
            // `CreateDXGIFactory2` if the debug interface is actually available. So
            // we check for whether it exists first.
            match lib_dxgi.get_debug_interface1() {
                Ok(pair) => match pair.check() {
                    Ok(debug_controller) => {
                        debug_controller.destroy();
                        factory_flags |= native::FactoryCreationFlags::DEBUG;
                    }
                    Err(err) => {
                        log::warn!("Unable to enable DXGI debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for DXGI not found: {:?}", err);
                }
            }
        }

        // Create DXGI factory
        let factory = match lib_dxgi.create_factory2(factory_flags) {
            Ok(pair) => match pair.check() {
                Ok(factory) => factory,
                Err(err) => {
                    log::warn!("Failed to create DXGI factory: {}", err);
                    return Err(crate::InstanceError);
                }
            },
            Err(err) => {
                log::warn!("Factory creation function for DXGI not found: {:?}", err);
                return Err(crate::InstanceError);
            }
        };

        Ok(Self {
            factory,
            library: Arc::new(lib_main),
            lib_dxgi,
        })
    }

    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        match has_handle.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Windows(handle) => {
                Ok(Surface {
                    factory: self.factory,
                    wnd_handle: handle.hwnd as *mut _,
                    //presentation: None,
                })
            }
            _ => Err(crate::InstanceError),
        }
    }
    unsafe fn destroy_surface(&self, _surface: Surface) {
        // just drop
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        // Try to use high performance order by default (returns None on Windows < 1803)
        let factory6 = match self.factory.cast::<dxgi1_6::IDXGIFactory6>().check() {
            Ok(f6) => {
                // It's okay to decrement the refcount here because we
                // have another reference to the factory already owned by `self`.
                f6.destroy();
                Some(f6)
            }
            Err(err) => {
                log::info!("Failed to cast DXGI to 1.6: {}", err);
                None
            }
        };

        // Enumerate adapters
        let mut adapters = Vec::new();
        for cur_index in 0.. {
            let raw = match factory6 {
                Some(factory) => {
                    let mut adapter2 = native::WeakPtr::<dxgi1_2::IDXGIAdapter2>::null();
                    let hr = factory.EnumAdapterByGpuPreference(
                        cur_index,
                        dxgi1_6::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        &dxgi1_2::IDXGIAdapter2::uuidof(),
                        adapter2.mut_void(),
                    );

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Some(err) = hr.to_error() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    adapter2
                }
                None => {
                    let mut adapter1 = native::WeakPtr::<dxgi::IDXGIAdapter1>::null();
                    let hr = self
                        .factory
                        .EnumAdapters1(cur_index, adapter1.mut_void() as *mut *mut _);

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Some(err) = hr.to_error() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    match adapter1.cast::<dxgi1_2::IDXGIAdapter2>().check() {
                        Ok(adapter2) => {
                            adapter1.destroy();
                            adapter2
                        }
                        Err(err) => {
                            log::error!("Failed casting to Adapter2: {}", err);
                            break;
                        }
                    }
                }
            };

            adapters.extend(Adapter::expose(raw, &self.library));
        }
        adapters
    }
}

impl crate::Surface<Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &Device) {}

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        Ok(None)
    }
    unsafe fn discard_texture(&mut self, texture: Resource) {}
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&Resource],
        signal_fence: Option<(&mut Resource, crate::FenceValue)>,
    ) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        texture: Resource,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }
}

impl crate::Device<Api> for Device {
    unsafe fn exit(self) {}
    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_buffer(&self, buffer: Resource) {}
    unsafe fn map_buffer(
        &self,
        buffer: &Resource,
        range: crate::MemoryRange,
    ) -> DeviceResult<crate::BufferMapping> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn unmap_buffer(&self, buffer: &Resource) -> DeviceResult<()> {
        Ok(())
    }
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

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<Api>,
    ) -> DeviceResult<Encoder> {
        Ok(Encoder)
    }
    unsafe fn destroy_command_encoder(&self, encoder: Encoder) {}

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
        shader: crate::ShaderInput,
    ) -> Result<Resource, crate::ShaderError> {
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

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_query_set(&self, set: Resource) {}
    unsafe fn create_fence(&self) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_fence(&self, fence: Resource) {}
    unsafe fn get_fence_value(&self, fence: &Resource) -> DeviceResult<crate::FenceValue> {
        Ok(0)
    }
    unsafe fn wait(
        &self,
        fence: &Resource,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> DeviceResult<bool> {
        Ok(true)
    }

    unsafe fn start_capture(&self) -> bool {
        false
    }
    unsafe fn stop_capture(&self) {}
}

impl crate::CommandEncoder<Api> for Encoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {}
    unsafe fn end_encoding(&mut self) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn reset_all<I>(&mut self, command_buffers: I) {}

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

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &Resource, dst: &Resource, regions: T) {}

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &Resource,
        src_usage: crate::TextureUses,
        dst: &Resource,
        regions: T,
    ) {
    }

    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Resource, dst: &Resource, regions: T) {}

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Resource,
        src_usage: crate::TextureUses,
        dst: &Resource,
        regions: T,
    ) {
    }

    unsafe fn begin_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn end_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn write_timestamp(&mut self, set: &Resource, index: u32) {}
    unsafe fn reset_queries(&mut self, set: &Resource, range: Range<u32>) {}
    unsafe fn copy_query_results(
        &mut self,
        set: &Resource,
        range: Range<u32>,
        buffer: &Resource,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<Api>) {}
    unsafe fn end_render_pass(&mut self) {}

    unsafe fn set_bind_group(
        &mut self,
        layout: &Resource,
        index: u32,
        group: &Resource,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
    }
    unsafe fn set_push_constants(
        &mut self,
        layout: &Resource,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u32],
    ) {
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {}
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {}
    unsafe fn end_debug_marker(&mut self) {}

    unsafe fn set_render_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, Api>,
        format: wgt::IndexFormat,
    ) {
    }
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: crate::BufferBinding<'a, Api>) {
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {}
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {}
    unsafe fn set_stencil_reference(&mut self, value: u32) {}
    unsafe fn set_blend_constants(&mut self, color: &wgt::Color) {}

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
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &Resource,
        offset: wgt::BufferAddress,
        count_buffer: &Resource,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Resource,
        offset: wgt::BufferAddress,
        count_buffer: &Resource,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {}
    unsafe fn end_compute_pass(&mut self) {}

    unsafe fn set_compute_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_indirect(&mut self, buffer: &Resource, offset: wgt::BufferAddress) {}
}
