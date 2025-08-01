/*!
# Vulkan API internals.

## Stack memory

Ash expects slices, which we don't generally have available.
We cope with this requirement by the combination of the following ways:
  - temporarily allocating `Vec` on heap, where overhead is permitted
  - growing temporary local storage

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
mod drm;
mod instance;
mod sampler;
mod semaphore_list;

pub use adapter::PhysicalDeviceFeatures;

use alloc::{boxed::Box, ffi::CString, sync::Arc, vec::Vec};
use core::{borrow::Borrow, ffi::CStr, fmt, marker::PhantomData, mem, num::NonZeroU32};

use arrayvec::ArrayVec;
use ash::{ext, khr, vk};
use bytemuck::{Pod, Zeroable};
use hashbrown::HashSet;
use parking_lot::{Mutex, RwLock};

use naga::FastHashMap;
use wgt::InternalCounter;

use semaphore_list::SemaphoreList;

const MILLIS_TO_NANOS: u64 = 1_000_000;
const MAX_TOTAL_ATTACHMENTS: usize = crate::MAX_COLOR_ATTACHMENTS * 2 + 1;

#[derive(Clone, Debug)]
pub struct Api;

impl crate::Api for Api {
    const VARIANT: wgt::Backend = wgt::Backend::Vulkan;

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
    type AccelerationStructure = AccelerationStructure;
    type PipelineCache = PipelineCache;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = ShaderModule;
    type RenderPipeline = RenderPipeline;
    type ComputePipeline = ComputePipeline;
}

crate::impl_dyn_resource!(
    Adapter,
    AccelerationStructure,
    BindGroup,
    BindGroupLayout,
    Buffer,
    CommandBuffer,
    CommandEncoder,
    ComputePipeline,
    Device,
    Fence,
    Instance,
    PipelineCache,
    PipelineLayout,
    QuerySet,
    Queue,
    RenderPipeline,
    Sampler,
    ShaderModule,
    Surface,
    SurfaceTexture,
    Texture,
    TextureView
);

struct DebugUtils {
    extension: ext::debug_utils::Instance,
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

#[derive(Debug)]
/// The properties related to the validation layer needed for the
/// DebugUtilsMessenger for their workarounds
struct ValidationLayerProperties {
    /// Validation layer description, from `vk::LayerProperties`.
    layer_description: CString,

    /// Validation layer specification version, from `vk::LayerProperties`.
    layer_spec_version: u32,
}

/// User data needed by `instance::debug_utils_messenger_callback`.
///
/// When we create the [`vk::DebugUtilsMessengerEXT`], the `pUserData`
/// pointer refers to one of these values.
#[derive(Debug)]
pub struct DebugUtilsMessengerUserData {
    /// The properties related to the validation layer, if present
    validation_layer_properties: Option<ValidationLayerProperties>,

    /// If the OBS layer is present. OBS never increments the version of their layer,
    /// so there's no reason to have the version.
    has_obs_layer: bool,
}

pub struct InstanceShared {
    raw: ash::Instance,
    extensions: Vec<&'static CStr>,
    drop_guard: Option<crate::DropGuard>,
    flags: wgt::InstanceFlags,
    memory_budget_thresholds: wgt::MemoryBudgetThresholds,
    debug_utils: Option<DebugUtils>,
    get_physical_device_properties: Option<khr::get_physical_device_properties2::Instance>,
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

/// Semaphore used to acquire a swapchain image.
#[derive(Debug)]
struct SwapchainAcquireSemaphore {
    /// A semaphore that is signaled when this image is safe for us to modify.
    ///
    /// When [`vkAcquireNextImageKHR`] returns the index of the next swapchain
    /// image that we should use, that image may actually still be in use by the
    /// presentation engine, and is not yet safe to modify. However, that
    /// function does accept a semaphore that it will signal when the image is
    /// indeed safe to begin messing with.
    ///
    /// This semaphore is:
    ///
    /// - waited for by the first queue submission to operate on this image
    ///   since it was acquired, and
    ///
    /// - signaled by [`vkAcquireNextImageKHR`] when the acquired image is ready
    ///   for us to use.
    ///
    /// [`vkAcquireNextImageKHR`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkAcquireNextImageKHR
    acquire: vk::Semaphore,

    /// True if the next command submission operating on this image should wait
    /// for [`acquire`].
    ///
    /// We must wait for `acquire` before drawing to this swapchain image, but
    /// because `wgpu-hal` queue submissions are always strongly ordered, only
    /// the first submission that works with a swapchain image actually needs to
    /// wait. We set this flag when this image is acquired, and clear it the
    /// first time it's passed to [`Queue::submit`] as a surface texture.
    ///
    /// Additionally, semaphores can only be waited on once, so we need to ensure
    /// that we only actually pass this semaphore to the first submission that
    /// uses that image.
    ///
    /// [`acquire`]: SwapchainAcquireSemaphore::acquire
    /// [`Queue::submit`]: crate::Queue::submit
    should_wait_for_acquire: bool,

    /// The fence value of the last command submission that wrote to this image.
    ///
    /// The next time we try to acquire this image, we'll block until
    /// this submission finishes, proving that [`acquire`] is ready to
    /// pass to `vkAcquireNextImageKHR` again.
    ///
    /// [`acquire`]: SwapchainAcquireSemaphore::acquire
    previously_used_submission_index: crate::FenceValue,
}

impl SwapchainAcquireSemaphore {
    fn new(device: &DeviceShared, index: usize) -> Result<Self, crate::DeviceError> {
        Ok(Self {
            acquire: device
                .new_binary_semaphore(&format!("SwapchainImageSemaphore: Index {index} acquire"))?,
            should_wait_for_acquire: true,
            previously_used_submission_index: 0,
        })
    }

    /// Sets the fence value which the next acquire will wait for. This prevents
    /// the semaphore from being used while the previous submission is still in flight.
    fn set_used_fence_value(&mut self, value: crate::FenceValue) {
        self.previously_used_submission_index = value;
    }

    /// Return the semaphore that commands drawing to this image should wait for, if any.
    ///
    /// This only returns `Some` once per acquisition; see
    /// [`SwapchainAcquireSemaphore::should_wait_for_acquire`] for details.
    fn get_acquire_wait_semaphore(&mut self) -> Option<vk::Semaphore> {
        if self.should_wait_for_acquire {
            self.should_wait_for_acquire = false;
            Some(self.acquire)
        } else {
            None
        }
    }

    /// Indicates the cpu-side usage of this semaphore has finished for the frame,
    /// so reset internal state to be ready for the next frame.
    fn end_semaphore_usage(&mut self) {
        // Reset the acquire semaphore, so that the next time we acquire this
        // image, we can wait for it again.
        self.should_wait_for_acquire = true;
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.acquire, None);
        }
    }
}

#[derive(Debug)]
struct SwapchainPresentSemaphores {
    /// A pool of semaphores for ordering presentation after drawing.
    ///
    /// The first [`present_index`] semaphores in this vector are:
    ///
    /// - all waited on by the call to [`vkQueuePresentKHR`] that presents this
    ///   image, and
    ///
    /// - each signaled by some [`vkQueueSubmit`] queue submission that draws to
    ///   this image, when the submission finishes execution.
    ///
    /// This vector accumulates one semaphore per submission that writes to this
    /// image. This is awkward, but hard to avoid: [`vkQueuePresentKHR`]
    /// requires a semaphore to order it with respect to drawing commands, and
    /// we can't attach new completion semaphores to a command submission after
    /// it's been submitted. This means that, at submission time, we must create
    /// the semaphore we might need if the caller's next action is to enqueue a
    /// presentation of this image.
    ///
    /// An alternative strategy would be for presentation to enqueue an empty
    /// submit, ordered relative to other submits in the usual way, and
    /// signaling a single presentation semaphore. But we suspect that submits
    /// are usually expensive enough, and semaphores usually cheap enough, that
    /// performance-sensitive users will avoid making many submits, so that the
    /// cost of accumulated semaphores will usually be less than the cost of an
    /// additional submit.
    ///
    /// Only the first [`present_index`] semaphores in the vector are actually
    /// going to be signalled by submitted commands, and need to be waited for
    /// by the next present call. Any semaphores beyond that index were created
    /// for prior presents and are simply being retained for recycling.
    ///
    /// [`present_index`]: SwapchainPresentSemaphores::present_index
    /// [`vkQueuePresentKHR`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkQueuePresentKHR
    /// [`vkQueueSubmit`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkQueueSubmit
    present: Vec<vk::Semaphore>,

    /// The number of semaphores in [`present`] to be signalled for this submission.
    ///
    /// [`present`]: SwapchainPresentSemaphores::present
    present_index: usize,

    /// Which image this semaphore set is used for.
    frame_index: usize,
}

impl SwapchainPresentSemaphores {
    pub fn new(frame_index: usize) -> Self {
        Self {
            present: Vec::new(),
            present_index: 0,
            frame_index,
        }
    }

    /// Return the semaphore that the next submission that writes to this image should
    /// signal when it's done.
    ///
    /// See [`SwapchainPresentSemaphores::present`] for details.
    fn get_submit_signal_semaphore(
        &mut self,
        device: &DeviceShared,
    ) -> Result<vk::Semaphore, crate::DeviceError> {
        // Try to recycle a semaphore we created for a previous presentation.
        let sem = match self.present.get(self.present_index) {
            Some(sem) => *sem,
            None => {
                let sem = device.new_binary_semaphore(&format!(
                    "SwapchainImageSemaphore: Image {} present semaphore {}",
                    self.frame_index, self.present_index
                ))?;
                self.present.push(sem);
                sem
            }
        };

        self.present_index += 1;

        Ok(sem)
    }

    /// Indicates the cpu-side usage of this semaphore has finished for the frame,
    /// so reset internal state to be ready for the next frame.
    fn end_semaphore_usage(&mut self) {
        // Reset the index to 0, so that the next time we get a semaphore, we
        // start from the beginning of the list.
        self.present_index = 0;
    }

    /// Return the semaphores that a presentation of this image should wait on.
    ///
    /// Return a slice of semaphores that the call to [`vkQueueSubmit`] that
    /// ends this image's acquisition should wait for. See
    /// [`SwapchainPresentSemaphores::present`] for details.
    ///
    /// Reset `self` to be ready for the next acquisition cycle.
    ///
    /// [`vkQueueSubmit`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkQueueSubmit
    fn get_present_wait_semaphores(&mut self) -> Vec<vk::Semaphore> {
        self.present[0..self.present_index].to_vec()
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            for sem in &self.present {
                device.destroy_semaphore(*sem, None);
            }
        }
    }
}

struct Swapchain {
    raw: vk::SwapchainKHR,
    functor: khr::swapchain::Device,
    device: Arc<DeviceShared>,
    images: Vec<vk::Image>,
    config: crate::SurfaceConfiguration,

    /// Semaphores used between image acquisition and the first submission
    /// that uses that image. This is indexed using [`next_acquire_index`].
    ///
    /// Because we need to provide this to [`vkAcquireNextImageKHR`], we haven't
    /// received the swapchain image index for the frame yet, so we cannot use
    /// that to index it.
    ///
    /// Before we pass this to [`vkAcquireNextImageKHR`], we ensure that we wait on
    /// the submission indicated by [`previously_used_submission_index`]. This enusres
    /// the semaphore is no longer in use before we use it.
    ///
    /// [`next_acquire_index`]: Swapchain::next_acquire_index
    /// [`vkAcquireNextImageKHR`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkAcquireNextImageKHR
    /// [`previously_used_submission_index`]: SwapchainAcquireSemaphore::previously_used_submission_index
    acquire_semaphores: Vec<Arc<Mutex<SwapchainAcquireSemaphore>>>,
    /// The index of the next acquire semaphore to use.
    ///
    /// This is incremented each time we acquire a new image, and wraps around
    /// to 0 when it reaches the end of [`acquire_semaphores`].
    ///
    /// [`acquire_semaphores`]: Swapchain::acquire_semaphores
    next_acquire_index: usize,

    /// Semaphore sets used between all submissions that write to an image and
    /// the presentation of that image.
    ///
    /// This is indexed by the swapchain image index returned by
    /// [`vkAcquireNextImageKHR`].
    ///
    /// We know it is safe to use these semaphores because use them
    /// _after_ the acquire semaphore. Because the acquire semaphore
    /// has been signaled, the previous presentation using that image
    /// is known-finished, so this semaphore is no longer in use.
    ///
    /// [`vkAcquireNextImageKHR`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkAcquireNextImageKHR
    present_semaphores: Vec<Arc<Mutex<SwapchainPresentSemaphores>>>,

    /// The present timing information which will be set in the next call to [`present()`](crate::Queue::present()).
    ///
    /// # Safety
    ///
    /// This must only be set if [`wgt::Features::VULKAN_GOOGLE_DISPLAY_TIMING`] is enabled, and
    /// so the VK_GOOGLE_display_timing extension is present.
    next_present_time: Option<vk::PresentTimeGOOGLE>,
}

impl Swapchain {
    /// Mark the current frame finished, advancing to the next acquire semaphore.
    fn advance_acquire_semaphore(&mut self) {
        let semaphore_count = self.acquire_semaphores.len();
        self.next_acquire_index = (self.next_acquire_index + 1) % semaphore_count;
    }

    /// Get the next acquire semaphore that should be used with this swapchain.
    fn get_acquire_semaphore(&self) -> Arc<Mutex<SwapchainAcquireSemaphore>> {
        self.acquire_semaphores[self.next_acquire_index].clone()
    }

    /// Get the set of present semaphores that should be used with the given image index.
    fn get_present_semaphores(&self, index: u32) -> Arc<Mutex<SwapchainPresentSemaphores>> {
        self.present_semaphores[index as usize].clone()
    }
}

pub struct Surface {
    raw: vk::SurfaceKHR,
    functor: khr::surface::Instance,
    instance: Arc<InstanceShared>,
    swapchain: RwLock<Option<Swapchain>>,
}

impl Surface {
    /// Get the raw Vulkan swapchain associated with this surface.
    ///
    /// Returns [`None`] if the surface is not configured.
    pub fn raw_swapchain(&self) -> Option<vk::SwapchainKHR> {
        let read = self.swapchain.read();
        read.as_ref().map(|it| it.raw)
    }

    /// Set the present timing information which will be used for the next [presentation](crate::Queue::present()) of this surface,
    /// using [VK_GOOGLE_display_timing].
    ///
    /// This can be used to give an id to presentations, for future use of [`vk::PastPresentationTimingGOOGLE`].
    /// Note that `wgpu-hal` does *not* provide a way to use that API - you should manually access this through [`ash`].
    ///
    /// This can also be used to add a "not before" timestamp to the presentation.
    ///
    /// The exact semantics of the fields are also documented in the [specification](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPresentTimeGOOGLE.html) for the extension.
    ///
    /// # Panics
    ///
    /// - If the surface hasn't been configured.
    /// - If the device doesn't [support present timing](wgt::Features::VULKAN_GOOGLE_DISPLAY_TIMING).
    ///
    /// [VK_GOOGLE_display_timing]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_display_timing.html
    #[track_caller]
    pub fn set_next_present_time(&self, present_timing: vk::PresentTimeGOOGLE) {
        let mut swapchain = self.swapchain.write();
        let swapchain = swapchain
            .as_mut()
            .expect("Surface should have been configured");
        let features = wgt::Features::VULKAN_GOOGLE_DISPLAY_TIMING;
        if swapchain.device.features.contains(features) {
            swapchain.next_present_time = Some(present_timing);
        } else {
            // Ideally we'd use something like `device.required_features` here, but that's in `wgpu-core`, which we are a dependency of
            panic!(
                concat!(
                    "Tried to set display timing properties ",
                    "without the corresponding feature ({:?}) enabled."
                ),
                features
            );
        }
    }
}

#[derive(Debug)]
pub struct SurfaceTexture {
    index: u32,
    texture: Texture,
    acquire_semaphores: Arc<Mutex<SwapchainAcquireSemaphore>>,
    present_semaphores: Arc<Mutex<SwapchainPresentSemaphores>>,
}

impl crate::DynSurfaceTexture for SurfaceTexture {}

impl Borrow<Texture> for SurfaceTexture {
    fn borrow(&self) -> &Texture {
        &self.texture
    }
}

impl Borrow<dyn crate::DynTexture> for SurfaceTexture {
    fn borrow(&self) -> &dyn crate::DynTexture {
        &self.texture
    }
}

pub struct Adapter {
    raw: vk::PhysicalDevice,
    instance: Arc<InstanceShared>,
    //queue_families: Vec<vk::QueueFamilyProperties>,
    known_memory_flags: vk::MemoryPropertyFlags,
    phd_capabilities: adapter::PhysicalDeviceProperties,
    phd_features: PhysicalDeviceFeatures,
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
    debug_utils: Option<ext::debug_utils::Device>,
    draw_indirect_count: Option<khr::draw_indirect_count::Device>,
    timeline_semaphore: Option<ExtensionFn<khr::timeline_semaphore::Device>>,
    ray_tracing: Option<RayTracingDeviceExtensionFunctions>,
    mesh_shading: Option<ext::mesh_shader::Device>,
}

struct RayTracingDeviceExtensionFunctions {
    acceleration_structure: khr::acceleration_structure::Device,
    buffer_device_address: khr::buffer_device_address::Device,
}

/// Set of internal capabilities, which don't show up in the exposed
/// device geometry, but affect the code paths taken internally.
#[derive(Clone, Debug)]
struct PrivateCapabilities {
    image_view_usage: bool,
    timeline_semaphores: bool,
    texture_d24: bool,
    texture_d24_s8: bool,
    texture_s8: bool,
    /// Ability to present contents to any screen. Only needed to work around broken platform configurations.
    can_present: bool,
    non_coherent_map_mask: wgt::BufferAddress,

    /// True if this adapter advertises the [`robustBufferAccess`][vrba] feature.
    ///
    /// Note that Vulkan's `robustBufferAccess` is not sufficient to implement
    /// `wgpu_hal`'s guarantee that shaders will not access buffer contents via
    /// a given bindgroup binding outside that binding's [accessible
    /// region][ar]. Enabling `robustBufferAccess` does ensure that
    /// out-of-bounds reads and writes are not undefined behavior (that's good),
    /// but still permits out-of-bounds reads to return data from anywhere
    /// within the buffer, not just the accessible region.
    ///
    /// [ar]: ../struct.BufferBinding.html#accessible-region
    /// [vrba]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-robustBufferAccess
    robust_buffer_access: bool,

    robust_image_access: bool,

    /// True if this adapter supports the [`VK_EXT_robustness2`] extension's
    /// [`robustBufferAccess2`] feature.
    ///
    /// This is sufficient to implement `wgpu_hal`'s [required bounds-checking][ar] of
    /// shader accesses to buffer contents. If this feature is not available,
    /// this backend must have Naga inject bounds checks in the generated
    /// SPIR-V.
    ///
    /// [`VK_EXT_robustness2`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_robustness2.html
    /// [`robustBufferAccess2`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceRobustness2FeaturesEXT.html#features-robustBufferAccess2
    /// [ar]: ../struct.BufferBinding.html#accessible-region
    robust_buffer_access2: bool,

    robust_image_access2: bool,
    zero_initialize_workgroup_memory: bool,
    image_format_list: bool,
    maximum_samplers: u32,

    /// True if this adapter supports the [`VK_KHR_shader_integer_dot_product`] extension
    /// (promoted to Vulkan 1.3).
    ///
    /// This is used to generate optimized code for WGSL's `dot4{I, U}8Packed`.
    ///
    /// [`VK_KHR_shader_integer_dot_product`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_integer_dot_product.html
    shader_integer_dot_product: bool,

    /// True if this adapter supports 8-bit integers provided by the
    /// [`VK_KHR_shader_float16_int8`] extension (promoted to Vulkan 1.2).
    ///
    /// Allows shaders to declare the "Int8" capability. Note, however, that this
    /// feature alone allows the use of 8-bit integers "only in the `Private`,
    /// `Workgroup` (for non-Block variables), and `Function` storage classes"
    /// ([see spec]). To use 8-bit integers in the interface storage classes (e.g.,
    /// `StorageBuffer`), you also need to enable the corresponding feature in
    /// `VkPhysicalDevice8BitStorageFeatures` and declare the corresponding SPIR-V
    /// capability (e.g., `StorageBuffer8BitAccess`).
    ///
    /// [`VK_KHR_shader_float16_int8`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_shader_float16_int8.html
    /// [see spec]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceShaderFloat16Int8Features.html#extension-features-shaderInt8
    shader_int8: bool,
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

struct DeviceShared {
    raw: ash::Device,
    family_index: u32,
    queue_index: u32,
    raw_queue: vk::Queue,
    drop_guard: Option<crate::DropGuard>,
    instance: Arc<InstanceShared>,
    physical_device: vk::PhysicalDevice,
    enabled_extensions: Vec<&'static CStr>,
    extension_fns: DeviceExtensionFunctions,
    vendor_id: u32,
    pipeline_cache_validation_key: [u8; 16],
    timestamp_period: f32,
    private_caps: PrivateCapabilities,
    workarounds: Workarounds,
    features: wgt::Features,
    render_passes: Mutex<FastHashMap<RenderPassKey, vk::RenderPass>>,
    sampler_cache: Mutex<sampler::SamplerCache>,
    memory_allocations_counter: InternalCounter,

    /// Because we have cached framebuffers which are not deleted from until
    /// the device is destroyed, if the implementation of vulkan re-uses handles
    /// we need some way to differentiate between the old handle and the new handle.
    /// This factory allows us to have a dedicated identity value for each texture.
    texture_identity_factory: ResourceIdentityFactory<vk::Image>,
    /// As above, for texture views.
    texture_view_identity_factory: ResourceIdentityFactory<vk::ImageView>,
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        for &raw in self.render_passes.lock().values() {
            unsafe { self.raw.destroy_render_pass(raw, None) };
        }
        if self.drop_guard.is_none() {
            unsafe { self.raw.destroy_device(None) };
        }
    }
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
    counters: Arc<wgt::HalCounters>,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.mem_allocator.lock().cleanup(&*self.shared) };
        unsafe { self.desc_allocator.lock().cleanup(&*self.shared) };
    }
}

/// Semaphores for forcing queue submissions to run in order.
///
/// The [`wgpu_hal::Queue`] trait promises that if two calls to [`submit`] are
/// ordered, then the first submission will finish on the GPU before the second
/// submission begins. To get this behavior on Vulkan we need to pass semaphores
/// to [`vkQueueSubmit`] for the commands to wait on before beginning execution,
/// and to signal when their execution is done.
///
/// Normally this can be done with a single semaphore, waited on and then
/// signalled for each submission. At any given time there's exactly one
/// submission that would signal the semaphore, and exactly one waiting on it,
/// as Vulkan requires.
///
/// However, as of Oct 2021, bug [#5508] in the Mesa ANV drivers caused them to
/// hang if we use a single semaphore. The workaround is to alternate between
/// two semaphores. The bug has been fixed in Mesa, but we should probably keep
/// the workaround until, say, Oct 2026.
///
/// [`wgpu_hal::Queue`]: crate::Queue
/// [`submit`]: crate::Queue::submit
/// [`vkQueueSubmit`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#vkQueueSubmit
/// [#5508]: https://gitlab.freedesktop.org/mesa/mesa/-/issues/5508
#[derive(Clone)]
struct RelaySemaphores {
    /// The semaphore the next submission should wait on before beginning
    /// execution on the GPU. This is `None` for the first submission, which
    /// should not wait on anything at all.
    wait: Option<vk::Semaphore>,

    /// The semaphore the next submission should signal when it has finished
    /// execution on the GPU.
    signal: vk::Semaphore,
}

impl RelaySemaphores {
    fn new(device: &DeviceShared) -> Result<Self, crate::DeviceError> {
        Ok(Self {
            wait: None,
            signal: device.new_binary_semaphore("RelaySemaphores: 1")?,
        })
    }

    /// Advances the semaphores, returning the semaphores that should be used for a submission.
    fn advance(&mut self, device: &DeviceShared) -> Result<Self, crate::DeviceError> {
        let old = self.clone();

        // Build the state for the next submission.
        match self.wait {
            None => {
                // The `old` values describe the first submission to this queue.
                // The second submission should wait on `old.signal`, and then
                // signal a new semaphore which we'll create now.
                self.wait = Some(old.signal);
                self.signal = device.new_binary_semaphore("RelaySemaphores: 2")?;
            }
            Some(ref mut wait) => {
                // What this submission signals, the next should wait.
                mem::swap(wait, &mut self.signal);
            }
        };

        Ok(old)
    }

    /// Destroys the semaphores.
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            if let Some(wait) = self.wait {
                device.destroy_semaphore(wait, None);
            }
            device.destroy_semaphore(self.signal, None);
        }
    }
}

pub struct Queue {
    raw: vk::Queue,
    swapchain_fn: khr::swapchain::Device,
    device: Arc<DeviceShared>,
    family_index: u32,
    relay_semaphores: Mutex<RelaySemaphores>,
    signal_semaphores: Mutex<SemaphoreList>,
}

impl Queue {
    pub fn as_raw(&self) -> vk::Queue {
        self.raw
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { self.relay_semaphores.lock().destroy(&self.device.raw) };
    }
}
#[derive(Debug)]
enum BufferMemoryBacking {
    Managed(gpu_alloc::MemoryBlock<vk::DeviceMemory>),
    VulkanMemory {
        memory: vk::DeviceMemory,
        offset: u64,
        size: u64,
    },
}
impl BufferMemoryBacking {
    fn memory(&self) -> &vk::DeviceMemory {
        match self {
            Self::Managed(m) => m.memory(),
            Self::VulkanMemory { memory, .. } => memory,
        }
    }
    fn offset(&self) -> u64 {
        match self {
            Self::Managed(m) => m.offset(),
            Self::VulkanMemory { offset, .. } => *offset,
        }
    }
    fn size(&self) -> u64 {
        match self {
            Self::Managed(m) => m.size(),
            Self::VulkanMemory { size, .. } => *size,
        }
    }
}
#[derive(Debug)]
pub struct Buffer {
    raw: vk::Buffer,
    block: Option<Mutex<BufferMemoryBacking>>,
}
impl Buffer {
    /// # Safety
    ///
    /// - `vk_buffer`'s memory must be managed by the caller
    /// - Externally imported buffers can't be mapped by `wgpu`
    pub unsafe fn from_raw(vk_buffer: vk::Buffer) -> Self {
        Self {
            raw: vk_buffer,
            block: None,
        }
    }
    /// # Safety
    /// - We will use this buffer and the buffer's backing memory range as if we have exclusive ownership over it, until the wgpu resource is dropped and the wgpu-hal object is cleaned up
    /// - Externally imported buffers can't be mapped by `wgpu`
    /// - `offset` and `size` must be valid with the allocation of `memory`
    pub unsafe fn from_raw_managed(
        vk_buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            raw: vk_buffer,
            block: Some(Mutex::new(BufferMemoryBacking::VulkanMemory {
                memory,
                offset,
                size,
            })),
        }
    }
}

impl crate::DynBuffer for Buffer {}

#[derive(Debug)]
pub struct AccelerationStructure {
    raw: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    block: Mutex<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    compacted_size_query: Option<vk::QueryPool>,
}

impl crate::DynAccelerationStructure for AccelerationStructure {}

#[derive(Debug)]
pub struct Texture {
    raw: vk::Image,
    drop_guard: Option<crate::DropGuard>,
    external_memory: Option<vk::DeviceMemory>,
    block: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    format: wgt::TextureFormat,
    copy_size: crate::CopyExtent,
    identity: ResourceIdentity<vk::Image>,
}

impl crate::DynTexture for Texture {}

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
    raw_texture: vk::Image,
    raw: vk::ImageView,
    layers: NonZeroU32,
    format: wgt::TextureFormat,
    raw_format: vk::Format,
    base_mip_level: u32,
    dimension: wgt::TextureViewDimension,
    texture_identity: ResourceIdentity<vk::Image>,
    view_identity: ResourceIdentity<vk::ImageView>,
}

impl crate::DynTextureView for TextureView {}

impl TextureView {
    /// # Safety
    ///
    /// - The image view handle must not be manually destroyed
    pub unsafe fn raw_handle(&self) -> vk::ImageView {
        self.raw
    }

    /// Returns the raw texture view, along with its identity.
    fn identified_raw_view(&self) -> IdentifiedTextureView {
        IdentifiedTextureView {
            raw: self.raw,
            identity: self.view_identity,
        }
    }
}

#[derive(Debug)]
pub struct Sampler {
    raw: vk::Sampler,
    create_info: vk::SamplerCreateInfo<'static>,
}

impl crate::DynSampler for Sampler {}

#[derive(Debug)]
pub struct BindGroupLayout {
    raw: vk::DescriptorSetLayout,
    desc_count: gpu_descriptor::DescriptorTotalCount,
    types: Box<[(vk::DescriptorType, u32)]>,
    /// Map of binding index to size,
    binding_arrays: Vec<(u32, NonZeroU32)>,
}

impl crate::DynBindGroupLayout for BindGroupLayout {}

#[derive(Debug)]
pub struct PipelineLayout {
    raw: vk::PipelineLayout,
    binding_arrays: naga::back::spv::BindingMap,
}

impl crate::DynPipelineLayout for PipelineLayout {}

#[derive(Debug)]
pub struct BindGroup {
    set: gpu_descriptor::DescriptorSet<vk::DescriptorSet>,
}

impl crate::DynBindGroup for BindGroup {}

/// Miscellaneous allocation recycling pool for `CommandAllocator`.
#[derive(Default)]
struct Temp {
    marker: Vec<u8>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier<'static>>,
    image_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
}

impl Temp {
    fn clear(&mut self) {
        self.marker.clear();
        self.buffer_barriers.clear();
        self.image_barriers.clear();
    }

    fn make_c_str(&mut self, name: &str) -> &CStr {
        self.marker.clear();
        self.marker.extend_from_slice(name.as_bytes());
        self.marker.push(0);
        unsafe { CStr::from_bytes_with_nul_unchecked(&self.marker) }
    }
}

/// Generates unique IDs for each resource of type `T`.
///
/// Because vk handles are not permanently unique, this
/// provides a way to generate unique IDs for each resource.
struct ResourceIdentityFactory<T> {
    #[cfg(not(target_has_atomic = "64"))]
    next_id: Mutex<u64>,
    #[cfg(target_has_atomic = "64")]
    next_id: core::sync::atomic::AtomicU64,
    _phantom: PhantomData<T>,
}

impl<T> ResourceIdentityFactory<T> {
    fn new() -> Self {
        Self {
            #[cfg(not(target_has_atomic = "64"))]
            next_id: Mutex::new(0),
            #[cfg(target_has_atomic = "64")]
            next_id: core::sync::atomic::AtomicU64::new(0),
            _phantom: PhantomData,
        }
    }

    /// Returns a new unique ID for a resource of type `T`.
    fn next(&self) -> ResourceIdentity<T> {
        #[cfg(not(target_has_atomic = "64"))]
        {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            ResourceIdentity {
                id,
                _phantom: PhantomData,
            }
        }

        #[cfg(target_has_atomic = "64")]
        ResourceIdentity {
            id: self
                .next_id
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed),
            _phantom: PhantomData,
        }
    }
}

/// A unique identifier for a resource of type `T`.
///
/// This is used as a hashable key for resources, which
/// is permanently unique through the lifetime of the program.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
struct ResourceIdentity<T> {
    id: u64,
    _phantom: PhantomData<T>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct FramebufferKey {
    raw_pass: vk::RenderPass,
    /// Because this is used as a key in a hash map, we need to include the identity
    /// so that this hashes differently, even if the ImageView handles are the same
    /// between different views.
    attachment_identities: ArrayVec<ResourceIdentity<vk::ImageView>, { MAX_TOTAL_ATTACHMENTS }>,
    /// While this is redundant for calculating the hash, we need access to an array
    /// of all the raw ImageViews when we are creating the actual framebuffer,
    /// so we store this here.
    attachment_views: ArrayVec<vk::ImageView, { MAX_TOTAL_ATTACHMENTS }>,
    extent: wgt::Extent3d,
}

impl FramebufferKey {
    fn push_view(&mut self, view: IdentifiedTextureView) {
        self.attachment_identities.push(view.identity);
        self.attachment_views.push(view.raw);
    }
}

/// A texture view paired with its identity.
#[derive(Copy, Clone)]
struct IdentifiedTextureView {
    raw: vk::ImageView,
    identity: ResourceIdentity<vk::ImageView>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct TempTextureViewKey {
    texture: vk::Image,
    /// As this is used in a hashmap, we need to
    /// include the identity so that this hashes differently,
    /// even if the Image handles are the same between different images.
    texture_identity: ResourceIdentity<vk::Image>,
    format: vk::Format,
    mip_level: u32,
    depth_slice: u32,
}

pub struct CommandEncoder {
    raw: vk::CommandPool,
    device: Arc<DeviceShared>,

    /// The current command buffer, if `self` is in the ["recording"]
    /// state.
    ///
    /// ["recording"]: crate::CommandEncoder
    ///
    /// If non-`null`, the buffer is in the Vulkan "recording" state.
    active: vk::CommandBuffer,

    /// What kind of pass we are currently within: compute or render.
    bind_point: vk::PipelineBindPoint,

    /// Allocation recycling pool for this encoder.
    temp: Temp,

    /// A pool of available command buffers.
    ///
    /// These are all in the Vulkan "initial" state.
    free: Vec<vk::CommandBuffer>,

    /// A pool of discarded command buffers.
    ///
    /// These could be in any Vulkan state except "pending".
    discarded: Vec<vk::CommandBuffer>,

    /// If this is true, the active renderpass enabled a debug span,
    /// and needs to be disabled on renderpass close.
    rpass_debug_marker_active: bool,

    /// If set, the end of the next render/compute pass will write a timestamp at
    /// the given pool & location.
    end_of_pass_timer_query: Option<(vk::QueryPool, u32)>,

    framebuffers: FastHashMap<FramebufferKey, vk::Framebuffer>,
    temp_texture_views: FastHashMap<TempTextureViewKey, IdentifiedTextureView>,

    counters: Arc<wgt::HalCounters>,
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        // SAFETY:
        //
        // VUID-vkDestroyCommandPool-commandPool-00041: wgpu_hal requires that a
        // `CommandBuffer` must live until its execution is complete, and that a
        // `CommandBuffer` must not outlive the `CommandEncoder` that built it.
        // Thus, we know that none of our `CommandBuffers` are in the "pending"
        // state.
        //
        // The other VUIDs are pretty obvious.
        unsafe {
            // `vkDestroyCommandPool` also frees any command buffers allocated
            // from that pool, so there's no need to explicitly call
            // `vkFreeCommandBuffers` on `cmd_encoder`'s `free` and `discarded`
            // fields.
            self.device.raw.destroy_command_pool(self.raw, None);
        }

        for (_, fb) in self.framebuffers.drain() {
            unsafe { self.device.raw.destroy_framebuffer(fb, None) };
        }

        for (_, view) in self.temp_texture_views.drain() {
            unsafe { self.device.raw.destroy_image_view(view.raw, None) };
        }

        self.counters.command_encoders.sub(1);
    }
}

impl CommandEncoder {
    /// # Safety
    ///
    /// - The command buffer handle must not be manually destroyed
    pub unsafe fn raw_handle(&self) -> vk::CommandBuffer {
        self.active
    }
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

impl crate::DynCommandBuffer for CommandBuffer {}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ShaderModule {
    Raw(vk::ShaderModule),
    Intermediate {
        naga_shader: crate::NagaShader,
        runtime_checks: wgt::ShaderRuntimeChecks,
    },
}

impl crate::DynShaderModule for ShaderModule {}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: vk::Pipeline,
}

impl crate::DynRenderPipeline for RenderPipeline {}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: vk::Pipeline,
}

impl crate::DynComputePipeline for ComputePipeline {}

#[derive(Debug)]
pub struct PipelineCache {
    raw: vk::PipelineCache,
}

impl crate::DynPipelineCache for PipelineCache {}

#[derive(Debug)]
pub struct QuerySet {
    raw: vk::QueryPool,
}

impl crate::DynQuerySet for QuerySet {}

/// The [`Api::Fence`] type for [`vulkan::Api`].
///
/// This is an `enum` because there are two possible implementations of
/// `wgpu-hal` fences on Vulkan: Vulkan fences, which work on any version of
/// Vulkan, and Vulkan timeline semaphores, which are easier and cheaper but
/// require non-1.0 features.
///
/// [`Device::create_fence`] returns a [`TimelineSemaphore`] if
/// [`VK_KHR_timeline_semaphore`] is available and enabled, and a [`FencePool`]
/// otherwise.
///
/// [`Api::Fence`]: crate::Api::Fence
/// [`vulkan::Api`]: Api
/// [`Device::create_fence`]: crate::Device::create_fence
/// [`TimelineSemaphore`]: Fence::TimelineSemaphore
/// [`VK_KHR_timeline_semaphore`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_timeline_semaphore
/// [`FencePool`]: Fence::FencePool
#[derive(Debug)]
pub enum Fence {
    /// A Vulkan [timeline semaphore].
    ///
    /// These are simpler to use than Vulkan fences, since timeline semaphores
    /// work exactly the way [`wpgu_hal::Api::Fence`] is specified to work.
    ///
    /// [timeline semaphore]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-semaphores
    /// [`wpgu_hal::Api::Fence`]: crate::Api::Fence
    TimelineSemaphore(vk::Semaphore),

    /// A collection of Vulkan [fence]s, each associated with a [`FenceValue`].
    ///
    /// The effective [`FenceValue`] of this variant is the greater of
    /// `last_completed` and the maximum value associated with a signalled fence
    /// in `active`.
    ///
    /// Fences are available in all versions of Vulkan, but since they only have
    /// two states, "signaled" and "unsignaled", we need to use a separate fence
    /// for each queue submission we might want to wait for, and remember which
    /// [`FenceValue`] each one represents.
    ///
    /// [fence]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-fences
    /// [`FenceValue`]: crate::FenceValue
    FencePool {
        last_completed: crate::FenceValue,
        /// The pending fence values have to be ascending.
        active: Vec<(crate::FenceValue, vk::Fence)>,
        free: Vec<vk::Fence>,
    },
}

impl crate::DynFence for Fence {}

impl Fence {
    /// Return the highest [`FenceValue`] among the signalled fences in `active`.
    ///
    /// As an optimization, assume that we already know that the fence has
    /// reached `last_completed`, and don't bother checking fences whose values
    /// are less than that: those fences remain in the `active` array only
    /// because we haven't called `maintain` yet to clean them up.
    ///
    /// [`FenceValue`]: crate::FenceValue
    fn check_active(
        device: &ash::Device,
        mut last_completed: crate::FenceValue,
        active: &[(crate::FenceValue, vk::Fence)],
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        for &(value, raw) in active.iter() {
            unsafe {
                if value > last_completed
                    && device
                        .get_fence_status(raw)
                        .map_err(map_host_device_oom_and_lost_err)?
                {
                    last_completed = value;
                }
            }
        }
        Ok(last_completed)
    }

    /// Return the highest signalled [`FenceValue`] for `self`.
    ///
    /// [`FenceValue`]: crate::FenceValue
    fn get_latest(
        &self,
        device: &ash::Device,
        extension: Option<&ExtensionFn<khr::timeline_semaphore::Device>>,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        match *self {
            Self::TimelineSemaphore(raw) => unsafe {
                Ok(match *extension.unwrap() {
                    ExtensionFn::Extension(ref ext) => ext
                        .get_semaphore_counter_value(raw)
                        .map_err(map_host_device_oom_and_lost_err)?,
                    ExtensionFn::Promoted => device
                        .get_semaphore_counter_value(raw)
                        .map_err(map_host_device_oom_and_lost_err)?,
                })
            },
            Self::FencePool {
                last_completed,
                ref active,
                free: _,
            } => Self::check_active(device, last_completed, active),
        }
    }

    /// Trim the internal state of this [`Fence`].
    ///
    /// This function has no externally visible effect, but you should call it
    /// periodically to keep this fence's resource consumption under control.
    ///
    /// For fences using the [`FencePool`] implementation, this function
    /// recycles fences that have been signaled. If you don't call this,
    /// [`Queue::submit`] will just keep allocating a new Vulkan fence every
    /// time it's called.
    ///
    /// [`FencePool`]: Fence::FencePool
    /// [`Queue::submit`]: crate::Queue::submit
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
                    unsafe { device.reset_fences(&free[base_free..]) }
                        .map_err(map_device_oom_err)?
                }
                *last_completed = latest;
            }
        }
        Ok(())
    }
}

impl crate::Queue for Queue {
    type A = Api;

    unsafe fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        surface_textures: &[&SurfaceTexture],
        (signal_fence, signal_value): (&mut Fence, crate::FenceValue),
    ) -> Result<(), crate::DeviceError> {
        let mut fence_raw = vk::Fence::null();

        let mut wait_stage_masks = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut signal_semaphores = SemaphoreList::default();

        // Double check that the same swapchain image isn't being given to us multiple times,
        // as that will deadlock when we try to lock them all.
        debug_assert!(
            {
                let mut check = HashSet::with_capacity(surface_textures.len());
                // We compare the Arcs by pointer, as Eq isn't well defined for SurfaceSemaphores.
                for st in surface_textures {
                    check.insert(Arc::as_ptr(&st.acquire_semaphores) as usize);
                    check.insert(Arc::as_ptr(&st.present_semaphores) as usize);
                }
                check.len() == surface_textures.len() * 2
            },
            "More than one surface texture is being used from the same swapchain. This will cause a deadlock in release."
        );

        let locked_swapchain_semaphores = surface_textures
            .iter()
            .map(|st| {
                let acquire = st
                    .acquire_semaphores
                    .try_lock()
                    .expect("Failed to lock surface acquire semaphore");
                let present = st
                    .present_semaphores
                    .try_lock()
                    .expect("Failed to lock surface present semaphore");

                (acquire, present)
            })
            .collect::<Vec<_>>();

        for (mut acquire_semaphore, mut present_semaphores) in locked_swapchain_semaphores {
            acquire_semaphore.set_used_fence_value(signal_value);

            // If we're the first submission to operate on this image, wait on
            // its acquire semaphore, to make sure the presentation engine is
            // done with it.
            if let Some(sem) = acquire_semaphore.get_acquire_wait_semaphore() {
                wait_stage_masks.push(vk::PipelineStageFlags::TOP_OF_PIPE);
                wait_semaphores.push(sem);
            }

            // Get a semaphore to signal when we're done writing to this surface
            // image. Presentation of this image will wait for this.
            let signal_semaphore = present_semaphores.get_submit_signal_semaphore(&self.device)?;
            signal_semaphores.push_binary(signal_semaphore);
        }

        let mut guard = self.signal_semaphores.lock();
        if !guard.is_empty() {
            signal_semaphores.append(&mut guard);
        }

        // In order for submissions to be strictly ordered, we encode a dependency between each submission
        // using a pair of semaphores. This adds a wait if it is needed, and signals the next semaphore.
        let semaphore_state = self.relay_semaphores.lock().advance(&self.device)?;

        if let Some(sem) = semaphore_state.wait {
            wait_stage_masks.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            wait_semaphores.push(sem);
        }

        signal_semaphores.push_binary(semaphore_state.signal);

        // We need to signal our wgpu::Fence if we have one, this adds it to the signal list.
        signal_fence.maintain(&self.device.raw)?;
        match *signal_fence {
            Fence::TimelineSemaphore(raw) => {
                signal_semaphores.push_timeline(raw, signal_value);
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
                            .create_fence(&vk::FenceCreateInfo::default(), None)
                            .map_err(map_host_device_oom_err)?
                    },
                };
                active.push((signal_value, fence_raw));
            }
        }

        let vk_cmd_buffers = command_buffers
            .iter()
            .map(|cmd| cmd.raw)
            .collect::<Vec<_>>();

        let mut vk_info = vk::SubmitInfo::default().command_buffers(&vk_cmd_buffers);

        vk_info = vk_info
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks);

        let mut vk_timeline_info = mem::MaybeUninit::uninit();
        vk_info = signal_semaphores.add_to_submit(vk_info, &mut vk_timeline_info);

        profiling::scope!("vkQueueSubmit");
        unsafe {
            self.device
                .raw
                .queue_submit(self.raw, &[vk_info], fence_raw)
                .map_err(map_host_device_oom_and_lost_err)?
        };
        Ok(())
    }

    unsafe fn present(
        &self,
        surface: &Surface,
        texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let mut swapchain = surface.swapchain.write();
        let ssc = swapchain.as_mut().unwrap();
        let mut acquire_semaphore = texture.acquire_semaphores.lock();
        let mut present_semaphores = texture.present_semaphores.lock();

        let wait_semaphores = present_semaphores.get_present_wait_semaphores();

        // Reset the acquire and present semaphores internal state
        // to be ready for the next frame.
        //
        // We do this before the actual call to present to ensure that
        // even if this method errors and early outs, we have reset
        // the state for next frame.
        acquire_semaphore.end_semaphore_usage();
        present_semaphores.end_semaphore_usage();

        drop(acquire_semaphore);

        let swapchains = [ssc.raw];
        let image_indices = [texture.index];
        let vk_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&wait_semaphores);

        let mut display_timing;
        let present_times;
        let vk_info = if let Some(present_time) = ssc.next_present_time.take() {
            debug_assert!(
                ssc.device
                    .features
                    .contains(wgt::Features::VULKAN_GOOGLE_DISPLAY_TIMING),
                "`next_present_time` should only be set if `VULKAN_GOOGLE_DISPLAY_TIMING` is enabled"
            );
            present_times = [present_time];
            display_timing = vk::PresentTimesInfoGOOGLE::default().times(&present_times);
            // SAFETY: We know that VK_GOOGLE_display_timing is present because of the safety contract on `next_present_time`.
            vk_info.push_next(&mut display_timing)
        } else {
            vk_info
        };

        let suboptimal = {
            profiling::scope!("vkQueuePresentKHR");
            unsafe { self.swapchain_fn.queue_present(self.raw, &vk_info) }.map_err(|error| {
                match error {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => crate::SurfaceError::Outdated,
                    vk::Result::ERROR_SURFACE_LOST_KHR => crate::SurfaceError::Lost,
                    // We don't use VK_EXT_full_screen_exclusive
                    // VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
                    _ => map_host_device_oom_and_lost_err(error).into(),
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

impl Queue {
    pub fn raw_device(&self) -> &ash::Device {
        &self.device.raw
    }

    pub fn add_signal_semaphore(&self, semaphore: vk::Semaphore, semaphore_value: Option<u64>) {
        let mut guard = self.signal_semaphores.lock();
        if let Some(value) = semaphore_value {
            guard.push_timeline(semaphore, value);
        } else {
            guard.push_binary(semaphore);
        }
    }
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
/// - VK_ERROR_OUT_OF_DEVICE_MEMORY
fn map_host_device_oom_err(err: vk::Result) -> crate::DeviceError {
    match err {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            get_oom_err(err)
        }
        e => get_unexpected_err(e),
    }
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
/// - VK_ERROR_OUT_OF_DEVICE_MEMORY
/// - VK_ERROR_DEVICE_LOST
fn map_host_device_oom_and_lost_err(err: vk::Result) -> crate::DeviceError {
    match err {
        vk::Result::ERROR_DEVICE_LOST => get_lost_err(),
        other => map_host_device_oom_err(other),
    }
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
/// - VK_ERROR_OUT_OF_DEVICE_MEMORY
/// - VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR
fn map_host_device_oom_and_ioca_err(err: vk::Result) -> crate::DeviceError {
    // We don't use VK_KHR_buffer_device_address
    // VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR
    map_host_device_oom_err(err)
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
fn map_host_oom_err(err: vk::Result) -> crate::DeviceError {
    match err {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => get_oom_err(err),
        e => get_unexpected_err(e),
    }
}

/// Maps
///
/// - VK_ERROR_OUT_OF_DEVICE_MEMORY
fn map_device_oom_err(err: vk::Result) -> crate::DeviceError {
    match err {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => get_oom_err(err),
        e => get_unexpected_err(e),
    }
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
/// - VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR
fn map_host_oom_and_ioca_err(err: vk::Result) -> crate::DeviceError {
    // We don't use VK_KHR_buffer_device_address
    // VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR
    map_host_oom_err(err)
}

/// Maps
///
/// - VK_ERROR_OUT_OF_HOST_MEMORY
/// - VK_ERROR_OUT_OF_DEVICE_MEMORY
/// - VK_PIPELINE_COMPILE_REQUIRED_EXT
/// - VK_ERROR_INVALID_SHADER_NV
fn map_pipeline_err(err: vk::Result) -> crate::DeviceError {
    // We don't use VK_EXT_pipeline_creation_cache_control
    // VK_PIPELINE_COMPILE_REQUIRED_EXT
    // We don't use VK_NV_glsl_shader
    // VK_ERROR_INVALID_SHADER_NV
    map_host_device_oom_err(err)
}

/// Returns [`crate::DeviceError::Unexpected`] or panics if the `internal_error_panic`
/// feature flag is enabled.
fn get_unexpected_err(_err: vk::Result) -> crate::DeviceError {
    #[cfg(feature = "internal_error_panic")]
    panic!("Unexpected Vulkan error: {_err:?}");

    #[allow(unreachable_code)]
    crate::DeviceError::Unexpected
}

/// Returns [`crate::DeviceError::OutOfMemory`].
fn get_oom_err(_err: vk::Result) -> crate::DeviceError {
    crate::DeviceError::OutOfMemory
}

/// Returns [`crate::DeviceError::Lost`] or panics if the `device_lost_panic`
/// feature flag is enabled.
fn get_lost_err() -> crate::DeviceError {
    #[cfg(feature = "device_lost_panic")]
    panic!("Device lost");

    #[allow(unreachable_code)]
    crate::DeviceError::Lost
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct RawTlasInstance {
    transform: [f32; 12],
    custom_data_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

/// Arguments to the [`CreateDeviceCallback`].
pub struct CreateDeviceCallbackArgs<'arg, 'pnext, 'this>
where
    'this: 'pnext,
{
    /// The extensions to enable for the device. You must not remove anything from this list,
    /// but you may add to it.
    pub extensions: &'arg mut Vec<&'static CStr>,
    /// The physical device features to enable. You may enable features, but must not disable any.
    pub device_features: &'arg mut PhysicalDeviceFeatures,
    /// The queue create infos for the device. You may add or modify queue create infos as needed.
    pub queue_create_infos: &'arg mut Vec<vk::DeviceQueueCreateInfo<'pnext>>,
    /// The create info for the device. You may add or modify things in the pnext chain, but
    /// do not turn features off. Additionally, do not add things to the list of extensions,
    /// or to the feature set, as all changes to that member will be overwritten.
    pub create_info: &'arg mut vk::DeviceCreateInfo<'pnext>,
    /// We need to have `'this` in the struct, so we can declare that all lifetimes coming from
    /// captures in the closure will live longer (and hence satisfy) `'pnext`. However, we
    /// don't actually directly use `'this`
    _phantom: PhantomData<&'this ()>,
}

/// Callback to allow changing the vulkan device creation parameters.
///
/// # Safety:
/// - If you want to add extensions, add the to the `Vec<'static CStr>` not the create info,
///   as the create info value will be overwritten.
/// - Callback must not remove features.
/// - Callback must not change anything to what the instance does not support.
pub type CreateDeviceCallback<'this> =
    dyn for<'arg, 'pnext> FnOnce(CreateDeviceCallbackArgs<'arg, 'pnext, 'this>) + 'this;

/// Arguments to the [`CreateInstanceCallback`].
pub struct CreateInstanceCallbackArgs<'arg, 'pnext, 'this>
where
    'this: 'pnext,
{
    /// The extensions to enable for the instance. You must not remove anything from this list,
    /// but you may add to it.
    pub extensions: &'arg mut Vec<&'static CStr>,
    /// The create info for the instance. You may add or modify things in the pnext chain, but
    /// do not turn features off. Additionally, do not add things to the list of extensions,
    /// all changes to that member will be overwritten.
    pub create_info: &'arg mut vk::InstanceCreateInfo<'pnext>,
    /// Vulkan entry point.
    pub entry: &'arg ash::Entry,
    /// We need to have `'this` in the struct, so we can declare that all lifetimes coming from
    /// captures in the closure will live longer (and hence satisfy) `'pnext`. However, we
    /// don't actually directly use `'this`
    _phantom: PhantomData<&'this ()>,
}

/// Callback to allow changing the vulkan instance creation parameters.
///
/// # Safety:
/// - If you want to add extensions, add the to the `Vec<'static CStr>` not the create info,
///   as the create info value will be overwritten.
/// - Callback must not remove features.
/// - Callback must not change anything to what the instance does not support.
pub type CreateInstanceCallback<'this> =
    dyn for<'arg, 'pnext> FnOnce(CreateInstanceCallbackArgs<'arg, 'pnext, 'this>) + 'this;
