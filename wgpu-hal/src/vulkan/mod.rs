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

use std::{borrow::Borrow, ffi::CStr, fmt, num::NonZeroU32, sync::Arc};

use arrayvec::ArrayVec;
use ash::{
    extensions::{ext, khr},
    vk,
};
use parking_lot::Mutex;

const MILLIS_TO_NANOS: u64 = 1_000_000;
const MAX_TOTAL_ATTACHMENTS: usize = crate::MAX_COLOR_ATTACHMENTS * 2 + 1;

#[derive(Clone, Debug)]
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

    /// Owning pointer to the debug messenger callback user data.
    ///
    /// `InstanceShared::drop` destroys the debug messenger before
    /// dropping this, so the callback should never receive a dangling
    /// user data pointer.
    #[allow(dead_code)]
    callback_data: Box<DebugUtilsMessengerUserData>,
}

pub struct DebugUtilsCreateInfo {
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: Box<DebugUtilsMessengerUserData>,
}

/// User data needed by `instance::debug_utils_messenger_callback`.
///
/// When we create the [`vk::DebugUtilsMessengerEXT`], the `pUserData`
/// pointer refers to one of these values.
#[derive(Debug)]
pub struct DebugUtilsMessengerUserData {
    /// Validation layer description, from `vk::LayerProperties`.
    validation_layer_description: std::ffi::CString,

    /// Validation layer specification version, from `vk::LayerProperties`.
    validation_layer_spec_version: u32,

    /// If the OBS layer is present. OBS never increments the version of their layer,
    /// so there's no reason to have the version.
    has_obs_layer: bool,
}

pub struct InstanceShared {
    raw: ash::Instance,
    extensions: Vec<&'static CStr>,
    drop_guard: Option<crate::DropGuard>,
    flags: wgt::InstanceFlags,
    debug_utils: Option<DebugUtils>,
    get_physical_device_properties: Option<khr::GetPhysicalDeviceProperties2>,
    entry: ash::Entry,
    has_nv_optimus: bool,
    android_sdk_version: u32,
    /// The instance API version.
    ///
    /// Which is the version of Vulkan supported for instance-level functionality.
    ///
    /// It is associated with a `VkInstance` and its children,
    /// except for a `VkPhysicalDevice` and its children.
    instance_api_version: u32,
}

pub struct Instance {
    shared: Arc<InstanceShared>,
}

struct Swapchain {
    raw: vk::SwapchainKHR,
    raw_flags: vk::SwapchainCreateFlagsKHR,
    functor: khr::Swapchain,
    device: Arc<DeviceShared>,
    fence: vk::Fence,
    images: Vec<vk::Image>,
    config: crate::SurfaceConfiguration,
    view_formats: Vec<wgt::TextureFormat>,
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
    workarounds: Workarounds,
}

// TODO there's no reason why this can't be unified--the function pointers should all be the same--it's not clear how to do this with `ash`.
enum ExtensionFn<T> {
    /// The loaded function pointer struct for an extension.
    Extension(T),
    /// The extension was promoted to a core version of Vulkan and the functions on `ash`'s `DeviceV1_x` traits should be used.
    Promoted,
}

struct DeviceExtensionFunctions {
    draw_indirect_count: Option<khr::DrawIndirectCount>,
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
    texture_s8: bool,
    /// Ability to present contents to any screen. Only needed to work around broken platform configurations.
    can_present: bool,
    non_coherent_map_mask: wgt::BufferAddress,
    robust_buffer_access: bool,
    robust_image_access: bool,
    robust_buffer_access2: bool,
    robust_image_access2: bool,
    zero_initialize_workgroup_memory: bool,
    image_format_list: bool,
}

bitflags::bitflags!(
    /// Workaround flags.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct Workarounds: u32 {
        /// Only generate SPIR-V for one entry point at a time.
        const SEPARATE_ENTRY_POINTS = 0x1;
        /// Qualcomm OOMs when there are zero color attachments but a non-null pointer
        /// to a subpass resolve attachment array. This nulls out that pointer in that case.
        const EMPTY_RESOLVE_ATTACHMENT_LISTS = 0x2;
        /// If the following code returns false, then nvidia will end up filling the wrong range.
        ///
        /// ```skip
        /// fn nvidia_succeeds() -> bool {
        ///   # let (copy_length, start_offset) = (0, 0);
        ///     if copy_length >= 4096 {
        ///         if start_offset % 16 != 0 {
        ///             if copy_length == 4096 {
        ///                 return true;
        ///             }
        ///             if copy_length % 16 == 0 {
        ///                 return false;
        ///             }
        ///         }
        ///     }
        ///     true
        /// }
        /// ```
        ///
        /// As such, we need to make sure all calls to vkCmdFillBuffer are aligned to 16 bytes
        /// if they cover a range of 4096 bytes or more.
        const FORCE_FILL_BUFFER_WITH_SIZE_GREATER_4096_ALIGNED_OFFSET_16 = 0x4;
    }
);

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
    colors: ArrayVec<Option<ColorAttachmentKey>, { crate::MAX_COLOR_ATTACHMENTS }>,
    depth_stencil: Option<DepthStencilAttachmentKey>,
    sample_count: u32,
    multiview: Option<NonZeroU32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FramebufferAttachment {
    /// Can be NULL if the framebuffer is image-less
    raw: vk::ImageView,
    raw_image_flags: vk::ImageCreateFlags,
    view_usage: crate::TextureUses,
    view_format: wgt::TextureFormat,
    raw_view_formats: Vec<vk::Format>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct FramebufferKey {
    attachments: ArrayVec<FramebufferAttachment, { MAX_TOTAL_ATTACHMENTS }>,
    extent: wgt::Extent3d,
    sample_count: u32,
}

struct DeviceShared {
    raw: ash::Device,
    family_index: u32,
    queue_index: u32,
    raw_queue: ash::vk::Queue,
    handle_is_owned: bool,
    instance: Arc<InstanceShared>,
    physical_device: ash::vk::PhysicalDevice,
    enabled_extensions: Vec<&'static CStr>,
    extension_fns: DeviceExtensionFunctions,
    vendor_id: u32,
    timestamp_period: f32,
    private_caps: PrivateCapabilities,
    workarounds: Workarounds,
    render_passes: Mutex<rustc_hash::FxHashMap<RenderPassKey, vk::RenderPass>>,
    framebuffers: Mutex<rustc_hash::FxHashMap<FramebufferKey, vk::Framebuffer>>,
}

pub struct Device {
    shared: Arc<DeviceShared>,
    mem_allocator: Mutex<gpu_alloc::GpuAllocator<vk::DeviceMemory>>,
    desc_allocator:
        Mutex<gpu_descriptor::DescriptorAllocator<vk::DescriptorPool, vk::DescriptorSet>>,
    valid_ash_memory_types: u32,
    naga_options: naga::back::spv::Options<'static>,
    #[cfg(feature = "renderdoc")]
    render_doc: crate::auxil::renderdoc::RenderDoc,
}

pub struct Queue {
    raw: vk::Queue,
    swapchain_fn: khr::Swapchain,
    device: Arc<DeviceShared>,
    family_index: u32,
    /// We use a redundant chain of semaphores to pass on the signal
    /// from submissions to the last present, since it's required by the
    /// specification.
    /// It would be correct to use a single semaphore there, but
    /// [Intel hangs in `anv_queue_finish`](https://gitlab.freedesktop.org/mesa/mesa/-/issues/5508).
    relay_semaphores: [vk::Semaphore; 2],
    relay_index: Option<usize>,
}

#[derive(Debug)]
pub struct Buffer {
    raw: vk::Buffer,
    block: Option<Mutex<gpu_alloc::MemoryBlock<vk::DeviceMemory>>>,
}

#[derive(Debug)]
pub struct Texture {
    raw: vk::Image,
    drop_guard: Option<crate::DropGuard>,
    block: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    usage: crate::TextureUses,
    format: wgt::TextureFormat,
    raw_flags: vk::ImageCreateFlags,
    copy_size: crate::CopyExtent,
    view_formats: Vec<wgt::TextureFormat>,
}

impl Texture {
    /// # Safety
    ///
    /// - The image handle must not be manually destroyed
    pub unsafe fn raw_handle(&self) -> vk::Image {
        self.raw
    }
}

#[derive(Debug)]
pub struct TextureView {
    raw: vk::ImageView,
    layers: NonZeroU32,
    attachment: FramebufferAttachment,
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
    /// Map of binding index to size,
    binding_arrays: Vec<(u32, NonZeroU32)>,
}

#[derive(Debug)]
pub struct PipelineLayout {
    raw: vk::PipelineLayout,
    binding_arrays: naga::back::spv::BindingMap,
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
    /// If this is true, the active renderpass enabled a debug span,
    /// and needs to be disabled on renderpass close.
    rpass_debug_marker_active: bool,

    /// If set, the end of the next render/compute pass will write a timestamp at
    /// the given pool & location.
    end_of_pass_timer_query: Option<(vk::QueryPool, u32)>,
}

impl fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandEncoder")
            .field("raw", &self.raw)
            .finish()
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    raw: vk::CommandBuffer,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ShaderModule {
    Raw(vk::ShaderModule),
    Intermediate {
        naga_shader: crate::NagaShader,
        runtime_checks: bool,
    },
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
                    ExtensionFn::Extension(ref ext) => ext.get_semaphore_counter_value(raw)?,
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
        let mut signal_semaphores = [vk::Semaphore::null(), vk::Semaphore::null()];
        let signal_values;

        if let Some((fence, value)) = signal_fence {
            fence.maintain(&self.device.raw)?;
            match *fence {
                Fence::TimelineSemaphore(raw) => {
                    signal_values = [!0, value];
                    signal_semaphores[1] = raw;
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
                        None => unsafe {
                            self.device
                                .raw
                                .create_fence(&vk::FenceCreateInfo::builder(), None)?
                        },
                    };
                    active.push((value, fence_raw));
                }
            }
        }

        let wait_stage_mask = [vk::PipelineStageFlags::TOP_OF_PIPE];
        let sem_index = match self.relay_index {
            Some(old_index) => {
                vk_info = vk_info
                    .wait_semaphores(&self.relay_semaphores[old_index..old_index + 1])
                    .wait_dst_stage_mask(&wait_stage_mask);
                (old_index + 1) % self.relay_semaphores.len()
            }
            None => 0,
        };
        self.relay_index = Some(sem_index);
        signal_semaphores[0] = self.relay_semaphores[sem_index];

        let signal_count = if signal_semaphores[1] == vk::Semaphore::null() {
            1
        } else {
            2
        };
        vk_info = vk_info.signal_semaphores(&signal_semaphores[..signal_count]);

        profiling::scope!("vkQueueSubmit");
        unsafe {
            self.device
                .raw
                .queue_submit(self.raw, &[vk_info.build()], fence_raw)?
        };
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
        let mut vk_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        if let Some(old_index) = self.relay_index.take() {
            vk_info = vk_info.wait_semaphores(&self.relay_semaphores[old_index..old_index + 1]);
        }

        let suboptimal = {
            profiling::scope!("vkQueuePresentKHR");
            unsafe { self.swapchain_fn.queue_present(self.raw, &vk_info) }.map_err(|error| {
                match error {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => crate::SurfaceError::Outdated,
                    vk::Result::ERROR_SURFACE_LOST_KHR => crate::SurfaceError::Lost,
                    _ => crate::DeviceError::from(error).into(),
                }
            })?
        };
        if suboptimal {
            // We treat `VK_SUBOPTIMAL_KHR` as `VK_SUCCESS` on Android.
            // On Android 10+, libvulkan's `vkQueuePresentKHR` implementation returns `VK_SUBOPTIMAL_KHR` if not doing pre-rotation
            // (i.e `VkSwapchainCreateInfoKHR::preTransform` not being equal to the current device orientation).
            // This is always the case when the device orientation is anything other than the identity one, as we unconditionally use `VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR`.
            #[cfg(not(target_os = "android"))]
            log::warn!("Suboptimal present of frame {}", texture.index);
        }
        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        self.device.timestamp_period
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
