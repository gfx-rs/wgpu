/*!
# Vulkan API internals.

## Stack memory

Ash expects slices, which we don't generally have available.
We cope with this requirement by the combination of the following ways:
  - temporarily allocating `Vec` on heap, where overhead is permitted
  - growing temporary local storage
  - using `implace_it` on iterators

## Framebuffers and Render passes

Render passes are cached on the device and kept forever.

Framebuffers are also cached on the device, but they are removed when
any of the image views (they have) gets removed.
If Vulkan supports image-less framebuffers,
then the actual views are excluded from the framebuffer key.

## Fences

If timeline semaphores are available, they are used 1:1 with wgpu-hal fences.
Otherwise, we manage a pool of `VkFence` objects behind each `hal::Fence`.

!*/

mod adapter;
mod command;
mod conv;
mod device;
mod instance;

use std::{borrow::Borrow, ffi::CStr, sync::Arc};

use arrayvec::ArrayVec;
use ash::{
    extensions::{ext, khr},
    vk,
};
use parking_lot::Mutex;

const MILLIS_TO_NANOS: u64 = 1_000_000;
const MAX_TOTAL_ATTACHMENTS: usize = crate::MAX_COLOR_TARGETS * 2 + 1;

pub type DropGuard = Box<dyn std::any::Any + Send + Sync>;

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

struct DebugUtils {
    extension: ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

struct InstanceShared {
    raw: ash::Instance,
    drop_guard: Option<DropGuard>,
    flags: crate::InstanceFlags,
    debug_utils: Option<DebugUtils>,
    get_physical_device_properties: Option<vk::KhrGetPhysicalDeviceProperties2Fn>,
    entry: ash::Entry,
}

pub struct Instance {
    shared: Arc<InstanceShared>,
    extensions: Vec<&'static CStr>,
}

struct Swapchain {
    raw: vk::SwapchainKHR,
    functor: khr::Swapchain,
    device: Arc<DeviceShared>,
    fence: vk::Fence,
    images: Vec<vk::Image>,
    config: crate::SurfaceConfiguration,
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
    //queue_families: Vec<vk::QueueFamilyProperties>,
    known_memory_flags: vk::MemoryPropertyFlags,
    phd_capabilities: adapter::PhysicalDeviceCapabilities,
    //phd_features: adapter::PhysicalDeviceFeatures,
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
    timeline_semaphore: Option<ExtensionFn<khr::TimelineSemaphore>>,
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
    timeline_semaphores: bool,
    texture_d24: bool,
    texture_d24_s8: bool,
    non_coherent_map_mask: wgt::BufferAddress,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AttachmentKey {
    format: vk::Format,
    layout: vk::ImageLayout,
    ops: crate::AttachmentOps,
}

impl AttachmentKey {
    /// Returns an attachment key for a compatible attachment.
    fn compatible(format: vk::Format, layout: vk::ImageLayout) -> Self {
        Self {
            format,
            layout,
            ops: crate::AttachmentOps::all(),
        }
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct ColorAttachmentKey {
    base: AttachmentKey,
    resolve: Option<AttachmentKey>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct DepthStencilAttachmentKey {
    base: AttachmentKey,
    stencil_ops: crate::AttachmentOps,
}

#[derive(Clone, Eq, Default, Hash, PartialEq)]
struct RenderPassKey {
    colors: ArrayVec<ColorAttachmentKey, { crate::MAX_COLOR_TARGETS }>,
    depth_stencil: Option<DepthStencilAttachmentKey>,
    sample_count: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FramebufferAttachment {
    /// Can be NULL if the framebuffer is image-less
    raw: vk::ImageView,
    raw_image_flags: vk::ImageCreateFlags,
    view_usage: crate::TextureUses,
    view_format: wgt::TextureFormat,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct FramebufferKey {
    attachments: ArrayVec<FramebufferAttachment, { MAX_TOTAL_ATTACHMENTS }>,
    extent: wgt::Extent3d,
    sample_count: u32,
}

struct DeviceShared {
    raw: ash::Device,
    handle_is_owned: bool,
    instance: Arc<InstanceShared>,
    extension_fns: DeviceExtensionFunctions,
    vendor_id: u32,
    _timestamp_period: f32,
    downlevel_flags: wgt::DownlevelFlags,
    private_caps: PrivateCapabilities,
    render_passes: Mutex<fxhash::FxHashMap<RenderPassKey, vk::RenderPass>>,
    framebuffers: Mutex<fxhash::FxHashMap<FramebufferKey, vk::Framebuffer>>,
}

pub struct Device {
    shared: Arc<DeviceShared>,
    mem_allocator: Mutex<gpu_alloc::GpuAllocator<vk::DeviceMemory>>,
    desc_allocator:
        Mutex<gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>>,
    valid_ash_memory_types: u32,
    naga_options: naga::back::spv::Options,
    #[cfg(feature = "renderdoc")]
    render_doc: crate::auxil::renderdoc::RenderDoc,
}

pub struct Queue {
    raw: vk::Queue,
    swapchain_fn: khr::Swapchain,
    device: Arc<DeviceShared>,
    family_index: u32,
    /// This special semaphore is used to synchronize GPU work of
    /// everything on a queue with... itself. Yikes!
    /// It's required by the confusing portion of the spec to be signalled
    /// by last submission and waited by the present.
    relay_semaphore: vk::Semaphore,
    relay_active: bool,
}

#[derive(Debug)]
pub struct Buffer {
    raw: vk::Buffer,
    block: Mutex<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
}

#[derive(Debug)]
pub struct Texture {
    raw: vk::Image,
    drop_guard: Option<DropGuard>,
    block: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    usage: crate::TextureUses,
    aspects: crate::FormatAspects,
    format_info: wgt::TextureFormatInfo,
    raw_flags: vk::ImageCreateFlags,
}

#[derive(Debug)]
pub struct TextureView {
    raw: vk::ImageView,
    attachment: FramebufferAttachment,
}

impl TextureView {
    fn aspects(&self) -> crate::FormatAspects {
        self.attachment.view_format.into()
    }
}

#[derive(Debug)]
pub struct Sampler {
    raw: vk::Sampler,
}

#[derive(Debug)]
pub struct BindGroupLayout {
    raw: vk::DescriptorSetLayout,
    desc_count: gpu_descriptor::DescriptorTotalCount,
    types: Box<[(vk::DescriptorType, u32)]>,
}

#[derive(Debug)]
pub struct PipelineLayout {
    raw: vk::PipelineLayout,
}

#[derive(Debug)]
pub struct BindGroup {
    set: gpu_descriptor::DescriptorSet<vk::DescriptorSet>,
}

#[derive(Default)]
struct Temp {
    marker: Vec<u8>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier>,
    image_barriers: Vec<vk::ImageMemoryBarrier>,
}

unsafe impl Send for Temp {}
unsafe impl Sync for Temp {}

impl Temp {
    fn clear(&mut self) {
        self.marker.clear();
        self.buffer_barriers.clear();
        self.image_barriers.clear();
        //see also - https://github.com/NotIntMan/inplace_it/issues/8
    }

    fn make_c_str(&mut self, name: &str) -> &CStr {
        self.marker.clear();
        self.marker.extend_from_slice(name.as_bytes());
        self.marker.push(0);
        unsafe { CStr::from_bytes_with_nul_unchecked(&self.marker) }
    }
}

pub struct CommandEncoder {
    raw: vk::CommandPool,
    device: Arc<DeviceShared>,
    active: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    temp: Temp,
    free: Vec<vk::CommandBuffer>,
    discarded: Vec<vk::CommandBuffer>,
}

pub struct CommandBuffer {
    raw: vk::CommandBuffer,
}

#[derive(Debug)]
pub struct ShaderModule {
    raw: vk::ShaderModule,
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: vk::Pipeline,
}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: vk::Pipeline,
}

#[derive(Debug)]
pub struct QuerySet {
    raw: vk::QueryPool,
}

#[derive(Debug)]
pub enum Fence {
    TimelineSemaphore(vk::Semaphore),
    FencePool {
        last_completed: crate::FenceValue,
        /// The pending fence values have to be ascending.
        active: Vec<(crate::FenceValue, vk::Fence)>,
        free: Vec<vk::Fence>,
    },
}

impl Fence {
    fn check_active(
        device: &ash::Device,
        mut max_value: crate::FenceValue,
        active: &[(crate::FenceValue, vk::Fence)],
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        for &(value, raw) in active.iter() {
            unsafe {
                if value > max_value && device.get_fence_status(raw)? {
                    max_value = value;
                }
            }
        }
        Ok(max_value)
    }

    fn get_latest(
        &self,
        device: &ash::Device,
        extension: Option<&ExtensionFn<khr::TimelineSemaphore>>,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        match *self {
            Self::TimelineSemaphore(raw) => unsafe {
                Ok(match *extension.unwrap() {
                    ExtensionFn::Extension(ref ext) => {
                        ext.get_semaphore_counter_value(device.handle(), raw)?
                    }
                    ExtensionFn::Promoted => device.get_semaphore_counter_value(raw)?,
                })
            },
            Self::FencePool {
                last_completed,
                ref active,
                free: _,
            } => Self::check_active(device, last_completed, active),
        }
    }

    fn maintain(&mut self, device: &ash::Device) -> Result<(), crate::DeviceError> {
        match *self {
            Self::TimelineSemaphore(_) => {}
            Self::FencePool {
                ref mut last_completed,
                ref mut active,
                ref mut free,
            } => {
                let latest = Self::check_active(device, *last_completed, active)?;
                let base_free = free.len();
                for &(value, raw) in active.iter() {
                    if value <= latest {
                        free.push(raw);
                    }
                }
                if free.len() != base_free {
                    active.retain(|&(value, _)| value > latest);
                    unsafe {
                        device.reset_fences(&free[base_free..])?;
                    }
                }
                *last_completed = latest;
            }
        }
        Ok(())
    }
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&CommandBuffer],
        signal_fence: Option<(&mut Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        let vk_cmd_buffers = command_buffers
            .iter()
            .map(|cmd| cmd.raw)
            .collect::<Vec<_>>();

        let mut vk_info = vk::SubmitInfo::builder().command_buffers(&vk_cmd_buffers);

        let mut fence_raw = vk::Fence::null();
        let mut vk_timeline_info;
        let mut semaphores = [self.relay_semaphore, vk::Semaphore::null()];
        let signal_values;

        if let Some((fence, value)) = signal_fence {
            fence.maintain(&self.device.raw)?;
            match *fence {
                Fence::TimelineSemaphore(raw) => {
                    signal_values = [!0, value];
                    semaphores[1] = raw;
                    vk_timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
                        .signal_semaphore_values(&signal_values);
                    vk_info = vk_info.push_next(&mut vk_timeline_info);
                }
                Fence::FencePool {
                    ref mut active,
                    ref mut free,
                    ..
                } => {
                    fence_raw = match free.pop() {
                        Some(raw) => raw,
                        None => self
                            .device
                            .raw
                            .create_fence(&vk::FenceCreateInfo::builder(), None)?,
                    };
                    active.push((value, fence_raw));
                }
            }
        }

        let wait_stage_mask = [vk::PipelineStageFlags::TOP_OF_PIPE];
        if self.relay_active {
            vk_info = vk_info
                .wait_semaphores(&semaphores[..1])
                .wait_dst_stage_mask(&wait_stage_mask);
        }
        self.relay_active = true;
        let signal_count = if semaphores[1] == vk::Semaphore::null() {
            1
        } else {
            2
        };
        vk_info = vk_info.signal_semaphores(&semaphores[..signal_count]);

        self.device
            .raw
            .queue_submit(self.raw, &[vk_info.build()], fence_raw)?;
        Ok(())
    }

    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let ssc = surface.swapchain.as_ref().unwrap();

        let swapchains = [ssc.raw];
        let image_indices = [texture.index];
        let semaphores = [self.relay_semaphore];
        let mut vk_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        if self.relay_active {
            vk_info = vk_info.wait_semaphores(&semaphores);
            self.relay_active = false;
        }

        let suboptimal = self
            .swapchain_fn
            .queue_present(self.raw, &vk_info)
            .map_err(|error| match error {
                vk::Result::ERROR_OUT_OF_DATE_KHR => crate::SurfaceError::Outdated,
                vk::Result::ERROR_SURFACE_LOST_KHR => crate::SurfaceError::Lost,
                _ => crate::DeviceError::from(error).into(),
            })?;
        if suboptimal {
            log::warn!("Suboptimal present of frame {}", texture.index);
        }
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
