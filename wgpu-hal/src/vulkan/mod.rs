#![allow(unused_variables)]

mod adapter;
mod command;
mod conv;
mod device;
mod instance;

use std::{borrow::Borrow, ffi::CStr, sync::Arc};

use ash::{
    extensions::{ext, khr},
    vk,
};
use parking_lot::Mutex;

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

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = SurfaceTexture;
    type TextureView = TextureView;
    type Sampler = Sampler;
    type QuerySet = Resource;
    type Fence = Resource;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = Resource;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

struct DebugUtils {
    extension: ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

struct InstanceShared {
    raw: ash::Instance,
    flags: crate::InstanceFlag,
    debug_utils: Option<DebugUtils>,
    get_physical_device_properties: Option<vk::KhrGetPhysicalDeviceProperties2Fn>,
    //debug_messenger: Option<DebugMessenger>,
}

pub struct Instance {
    shared: Arc<InstanceShared>,
    extensions: Vec<&'static CStr>,
    entry: ash::Entry,
}

struct Swapchain {
    raw: vk::SwapchainKHR,
    functor: khr::Swapchain,
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
    downlevel_flags: wgt::DownlevelFlags,
    private_caps: PrivateCapabilities,
}

pub struct Device {
    shared: Arc<DeviceShared>,
    mem_allocator: Mutex<gpu_alloc::GpuAllocator<vk::DeviceMemory>>,
    valid_ash_memory_types: u32,
    naga_options: naga::back::spv::Options,
}

pub struct Queue {
    raw: vk::Queue,
    swapchain_fn: khr::Swapchain,
    //device: Arc<DeviceShared>,
}

#[derive(Debug)]
pub struct Buffer {
    raw: vk::Buffer,
    block: Mutex<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
}

#[derive(Debug)]
pub struct Texture {
    raw: vk::Image,
    block: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    aspects: crate::FormatAspect,
}

#[derive(Debug)]
pub struct TextureView {
    raw: vk::ImageView,
}

#[derive(Debug)]
pub struct Sampler {
    raw: vk::Sampler,
}

#[derive(Debug)]
pub struct BindGroupLayout {
    raw: vk::DescriptorSetLayout,
}

#[derive(Debug)]
pub struct PipelineLayout {
    raw: vk::PipelineLayout,
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
