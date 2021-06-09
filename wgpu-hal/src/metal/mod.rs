mod adapter;
mod command;
mod conv;
mod device;
mod surface;

use std::{iter, ops, ptr::NonNull, sync::Arc, thread};

use arrayvec::ArrayVec;
use foreign_types::ForeignTypeRef as _;
use parking_lot::Mutex;

#[derive(Clone)]
pub struct Api;
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

type ResourceIndex = u32;

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
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

pub struct Instance {}

impl crate::Instance<Api> for Instance {
    unsafe fn init() -> Result<Self, crate::InstanceError> {
        Ok(Instance {})
    }
    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        match has_handle.raw_window_handle() {
            #[cfg(target_os = "ios")]
            raw_window_handle::RawWindowHandle::IOS(handle) => {
                Ok(Surface::from_uiview(handle.ui_view))
            }
            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::MacOS(handle) => {
                Ok(Surface::from_nsview(handle.ns_view))
            }
            _ => Err(crate::InstanceError),
        }
    }

    unsafe fn destroy_surface(&self, surface: Surface) {}

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        let devices = mtl::Device::all();
        let mut adapters: Vec<crate::ExposedAdapter<Api>> = devices
            .into_iter()
            .map(|dev| {
                let name = dev.name().into();
                let shared = AdapterShared::new(dev);
                crate::ExposedAdapter {
                    info: wgt::AdapterInfo {
                        name,
                        vendor: 0,
                        device: 0,
                        device_type: if shared.private_caps.low_power {
                            wgt::DeviceType::IntegratedGpu
                        } else {
                            wgt::DeviceType::DiscreteGpu
                        },
                        backend: wgt::Backend::Metal,
                    },
                    features: shared.private_caps.features(),
                    capabilities: shared.private_caps.capabilities(),
                    adapter: Adapter::new(Arc::new(shared)),
                }
            })
            .collect();
        adapters.sort_by_key(|ad| {
            (
                ad.adapter.shared.private_caps.low_power,
                ad.adapter.shared.private_caps.headless,
            )
        });
        adapters
    }
}

#[derive(Clone, Debug)]
struct PrivateCapabilities {
    family_check: bool,
    msl_version: mtl::MTLLanguageVersion,
    exposed_queues: usize,
    read_write_texture_tier: mtl::MTLReadWriteTextureTier,
    resource_heaps: bool,
    argument_buffers: bool,
    shared_textures: bool,
    mutable_comparison_samplers: bool,
    sampler_clamp_to_border: bool,
    sampler_lod_average: bool,
    base_instance: bool,
    base_vertex_instance_drawing: bool,
    dual_source_blending: bool,
    low_power: bool,
    headless: bool,
    layered_rendering: bool,
    function_specialization: bool,
    depth_clip_mode: bool,
    texture_cube_array: bool,
    format_depth24_stencil8: bool,
    format_depth32_stencil8_filter: bool,
    format_depth32_stencil8_none: bool,
    format_min_srgb_channels: u8,
    format_b5: bool,
    format_bc: bool,
    format_eac_etc: bool,
    format_astc: bool,
    format_any8_unorm_srgb_all: bool,
    format_any8_unorm_srgb_no_write: bool,
    format_any8_snorm_all: bool,
    format_r16_norm_all: bool,
    format_r32_all: bool,
    format_r32_no_write: bool,
    format_r32float_no_write_no_filter: bool,
    format_r32float_no_filter: bool,
    format_r32float_all: bool,
    format_rgba8_srgb_all: bool,
    format_rgba8_srgb_no_write: bool,
    format_rgb10a2_unorm_all: bool,
    format_rgb10a2_unorm_no_write: bool,
    format_rgb10a2_uint_color: bool,
    format_rgb10a2_uint_color_write: bool,
    format_rg11b10_all: bool,
    format_rg11b10_no_write: bool,
    format_rgb9e5_all: bool,
    format_rgb9e5_no_write: bool,
    format_rgb9e5_filter_only: bool,
    format_rg32_color: bool,
    format_rg32_color_write: bool,
    format_rg32float_all: bool,
    format_rg32float_color_blend: bool,
    format_rg32float_no_filter: bool,
    format_rgba32int_color: bool,
    format_rgba32int_color_write: bool,
    format_rgba32float_color: bool,
    format_rgba32float_color_write: bool,
    format_rgba32float_all: bool,
    format_depth16unorm: bool,
    format_depth32float_filter: bool,
    format_depth32float_none: bool,
    format_bgr10a2_all: bool,
    format_bgr10a2_no_write: bool,
    max_buffers_per_stage: ResourceIndex,
    max_textures_per_stage: ResourceIndex,
    max_samplers_per_stage: ResourceIndex,
    buffer_alignment: u64,
    max_buffer_size: u64,
    max_texture_size: u64,
    max_texture_3d_size: u64,
    max_texture_layers: u64,
    max_fragment_input_components: u64,
    max_color_render_targets: u8,
    max_total_threadgroup_memory: u32,
    sample_count_mask: u8,
    supports_debug_markers: bool,
    supports_binary_archives: bool,
    can_set_maximum_drawables_count: bool,
    can_set_display_sync: bool,
    can_set_next_drawable_timeout: bool,
}

#[derive(Clone, Debug)]
struct PrivateDisabilities {
    /// Near depth is not respected properly on some Intel GPUs.
    broken_viewport_near_depth: bool,
    /// Multi-target clears don't appear to work properly on Intel GPUs.
    broken_layered_clear_image: bool,
}

struct AdapterShared {
    device: Mutex<mtl::Device>,
    disabilities: PrivateDisabilities,
    private_caps: PrivateCapabilities,
}

impl AdapterShared {
    fn new(device: mtl::Device) -> Self {
        let private_caps = PrivateCapabilities::new(&device);
        log::debug!("{:#?}", private_caps);

        Self {
            disabilities: PrivateDisabilities::new(&device),
            private_caps: PrivateCapabilities::new(&device),
            device: Mutex::new(device),
        }
    }
}

struct Adapter {
    shared: Arc<AdapterShared>,
}

struct Queue {
    raw: mtl::CommandQueue,
}

struct Device {
    shared: Arc<AdapterShared>,
    features: wgt::Features,
}

struct Surface {
    view: Option<NonNull<objc::runtime::Object>>,
    render_layer: Mutex<mtl::MetalLayer>,
    raw_swapchain_format: mtl::MTLPixelFormat,
    main_thread_id: thread::ThreadId,
    // Useful for UI-intensive applications that are sensitive to
    // window resizing.
    pub present_with_transaction: bool,
}

#[derive(Debug)]
struct SurfaceTexture {
    texture: Texture,
    drawable: mtl::MetalDrawable,
    present_with_transaction: bool,
}

impl std::borrow::Borrow<Texture> for SurfaceTexture {
    fn borrow(&self) -> &Texture {
        &self.texture
    }
}

unsafe impl Send for SurfaceTexture {}
unsafe impl Sync for SurfaceTexture {}

impl crate::Queue<Api> for Queue {
    unsafe fn submit<I>(
        &mut self,
        command_buffers: I,
        signal_fence: Option<(&Resource, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
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

#[derive(Debug)]
pub struct Buffer {
    raw: mtl::Buffer,
    size: wgt::BufferAddress,
    options: mtl::MTLResourceOptions,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn as_raw(&self) -> BufferPtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct Texture {
    raw: mtl::Texture,
    raw_format: mtl::MTLPixelFormat,
    raw_type: mtl::MTLTextureType,
    array_layers: u32,
    mip_levels: u32,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

#[derive(Debug)]
pub struct TextureView {
    raw: mtl::Texture,
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl TextureView {
    fn as_raw(&self) -> TexturePtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct Sampler {
    raw: mtl::SamplerState,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Sampler {
    fn as_raw(&self) -> SamplerPtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

type BindingMap = fxhash::FxHashMap<u32, wgt::BindGroupLayoutEntry>;

#[derive(Debug)]
pub struct BindGroupLayout {
    entries: Arc<BindingMap>,
}

#[derive(Clone, Debug, Default)]
struct ResourceData<T> {
    buffers: T,
    textures: T,
    samplers: T,
}

#[derive(Clone, Debug, Default)]
struct MultiStageData<T> {
    vs: T,
    fs: T,
    cs: T,
}

const NAGA_STAGES: MultiStageData<naga::ShaderStage> = MultiStageData {
    vs: naga::ShaderStage::Vertex,
    fs: naga::ShaderStage::Fragment,
    cs: naga::ShaderStage::Compute,
};

impl<T> ops::Index<naga::ShaderStage> for MultiStageData<T> {
    type Output = T;
    fn index(&self, stage: naga::ShaderStage) -> &T {
        match stage {
            naga::ShaderStage::Vertex => &self.vs,
            naga::ShaderStage::Fragment => &self.fs,
            naga::ShaderStage::Compute => &self.cs,
        }
    }
}

impl<T> MultiStageData<T> {
    fn map<Y>(&self, fun: impl Fn(&T) -> Y) -> MultiStageData<Y> {
        MultiStageData {
            vs: fun(&self.vs),
            fs: fun(&self.fs),
            cs: fun(&self.cs),
        }
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> {
        iter::once(&self.vs)
            .chain(iter::once(&self.fs))
            .chain(iter::once(&self.cs))
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        iter::once(&mut self.vs)
            .chain(iter::once(&mut self.fs))
            .chain(iter::once(&mut self.cs))
    }
}

type MultiStageResourceCounters = MultiStageData<ResourceData<ResourceIndex>>;

#[derive(Debug)]
struct BindGroupLayoutInfo {
    base_resource_indices: MultiStageResourceCounters,
    dynamic_buffers: Vec<MultiStageData<ResourceIndex>>,
    sized_buffer_bindings: Vec<(u32, wgt::ShaderStage)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PushConstantsStage {
    count: u32,
    buffer_index: ResourceIndex,
}

#[derive(Debug)]
pub struct PipelineLayout {
    naga_options: naga::back::msl::Options,
    bind_group_infos: ArrayVec<[BindGroupLayoutInfo; crate::MAX_BIND_GROUPS]>,
    push_constants_infos: MultiStageData<Option<PushConstantsStage>>,
}

trait AsNative {
    type Native;
    fn from(native: &Self::Native) -> Self;
    fn as_native(&self) -> &Self::Native;
}

type BufferPtr = NonNull<mtl::MTLBuffer>;
type TexturePtr = NonNull<mtl::MTLTexture>;
type SamplerPtr = NonNull<mtl::MTLSamplerState>;
type ResourcePtr = NonNull<mtl::MTLResource>;

impl AsNative for BufferPtr {
    type Native = mtl::BufferRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

impl AsNative for TexturePtr {
    type Native = mtl::TextureRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

impl AsNative for SamplerPtr {
    type Native = mtl::SamplerStateRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

#[derive(Debug)]
struct BufferResource {
    ptr: BufferPtr,
    offset: wgt::BufferAddress,
}

#[derive(Debug, Default)]
pub struct BindGroup {
    counters: MultiStageResourceCounters,
    buffers: Vec<BufferResource>,
    samplers: Vec<SamplerPtr>,
    textures: Vec<TexturePtr>,
}

unsafe impl Send for BindGroup {}
unsafe impl Sync for BindGroup {}
