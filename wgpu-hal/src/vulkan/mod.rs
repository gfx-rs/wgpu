#![allow(unused_variables)]

mod adapter;
mod conv;
mod device;
mod instance;

use ash::{extensions::khr, vk};

use std::{borrow::Borrow, ffi::CStr, ops::Range, sync::Arc};

const MILLIS_TO_NANOS: u64 = 1_000_000;

#[derive(Clone)]
pub struct Api;
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Queue = Queue;
    type Device = Device;

    type CommandBuffer = Encoder;

    type Buffer = Resource;
    type Texture = Texture;
    type SurfaceTexture = SurfaceTexture;
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

struct RenderDocEntry {
    api: renderdoc_sys::RENDERDOC_API_1_4_1,
    lib: libloading::Library,
}

unsafe impl Send for RenderDocEntry {}
unsafe impl Sync for RenderDocEntry {}

struct InstanceShared {
    raw: ash::Instance,
    flags: crate::InstanceFlag,
    get_physical_device_properties: Option<vk::KhrGetPhysicalDeviceProperties2Fn>,
    //TODO
    //debug_messenger: Option<DebugMessenger>,
    //render_doc_entry: Result<RenderDocEntry, String>,
}

pub struct Instance {
    shared: Arc<InstanceShared>,
    extensions: Vec<&'static CStr>,
    entry: ash::Entry,
}

struct Swapchain {
    raw: vk::SwapchainKHR,
    functor: khr::Swapchain,
    extent: vk::Extent3D,
    device: Arc<DeviceShared>,
    fence: vk::Fence,
    //semaphore: vk::Semaphore,
    images: Vec<vk::Image>,
}

pub struct Surface {
    raw: vk::SurfaceKHR,
    functor: khr::Surface,
    instance: Arc<InstanceShared>,
    swapchain: Option<Swapchain>,
}

#[derive(Debug)]
pub struct SurfaceTexture {
    index: u32,
    texture: Texture,
}

impl Borrow<Texture> for SurfaceTexture {
    fn borrow(&self) -> &Texture {
        &self.texture
    }
}

pub struct Adapter {
    raw: vk::PhysicalDevice,
    instance: Arc<InstanceShared>,
    queue_families: Vec<vk::QueueFamilyProperties>,
    known_memory_flags: vk::MemoryPropertyFlags,
    phd_capabilities: adapter::PhysicalDeviceCapabilities,
    phd_features: adapter::PhysicalDeviceFeatures,
    available_features: wgt::Features,
    downlevel_flags: wgt::DownlevelFlags,
    private_caps: PrivateCapabilities,
}

// TODO there's no reason why this can't be unified--the function pointers should all be the same--it's not clear how to do this with `ash`.
enum ExtensionFn<T> {
    /// The loaded function pointer struct for an extension.
    Extension(T),
    /// The extension was promoted to a core version of Vulkan and the functions on `ash`'s `DeviceV1_x` traits should be used.
    Promoted,
}

impl<T> ExtensionFn<T> {
    /// Expect `self` to be `Self::Extension` and return the inner value.
    fn unwrap_extension(&self) -> &T {
        match *self {
            Self::Extension(ref t) => t,
            Self::Promoted => panic!(),
        }
    }
}

struct DeviceExtensionFunctions {
    draw_indirect_count: Option<ExtensionFn<khr::DrawIndirectCount>>,
}

/// Set of internal capabilities, which don't show up in the exposed
/// device geometry, but affect the code paths taken internally.
#[derive(Clone, Debug)]
struct PrivateCapabilities {
    /// Y-flipping is implemented with either `VK_AMD_negative_viewport_height` or `VK_KHR_maintenance1`/1.1+. The AMD extension for negative viewport height does not require a Y shift.
    ///
    /// This flag is `true` if the device has `VK_KHR_maintenance1`/1.1+ and `false` otherwise (i.e. in the case of `VK_AMD_negative_viewport_height`).
    flip_y_requires_shift: bool,
    imageless_framebuffers: bool,
    image_view_usage: bool,
    texture_d24: bool,
    texture_d24_s8: bool,
}

struct DeviceShared {
    raw: ash::Device,
    instance: Arc<InstanceShared>,
    extension_fns: DeviceExtensionFunctions,
    features: wgt::Features,
    vendor_id: u32,
    timestamp_period: f32,
    private_caps: PrivateCapabilities,
}

pub struct Device {
    shared: Arc<DeviceShared>,
    valid_ash_memory_types: u32,
    naga_options: naga::back::spv::Options,
}

pub struct Queue {
    raw: vk::Queue,
    swapchain_fn: khr::Swapchain,
    //device: Arc<DeviceShared>,
}

#[derive(Debug)]
pub struct Texture {
    raw: vk::Image,
    ty: vk::ImageType,
    flags: vk::ImageCreateFlags,
    extent: vk::Extent3D,
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit<I>(
        &mut self,
        command_buffers: I,
        signal_fence: Option<(&mut Resource, crate::FenceValue)>,
    ) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }
}

pub struct Context;
impl crate::Device<Api> for Context {
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

    unsafe fn create_texture(&self, desc: &crate::TextureDescriptor) -> DeviceResult<Texture> {
        unimplemented!()
    }
    unsafe fn destroy_texture(&self, texture: Texture) {}
    unsafe fn create_texture_view(
        &self,
        texture: &Texture,
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

    unsafe fn create_query_set(&self, desc: &wgt::QuerySetDescriptor) -> DeviceResult<Resource> {
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

impl crate::CommandBuffer<Api> for Encoder {
    unsafe fn finish(&mut self) {}

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
        src: &Texture,
        src_usage: crate::TextureUse,
        dst: &Texture,
        regions: T,
    ) {
    }

    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Resource, dst: &Texture, regions: T) {}

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Texture,
        src_usage: crate::TextureUse,
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
        stages: wgt::ShaderStage,
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

impl From<vk::Result> for crate::DeviceError {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                Self::OutOfMemory
            }
            vk::Result::ERROR_DEVICE_LOST => Self::Lost,
            _ => {
                log::warn!("Unrecognized device error {:?}", result);
                Self::Lost
            }
        }
    }
}
