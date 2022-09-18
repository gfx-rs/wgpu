/*!
# Metal API internals.

## Pipeline Layout

In Metal, push constants, vertex buffers, and resources in the bind groups
are all placed together in the native resource bindings, which work similarly to D3D11:
there are tables of textures, buffers, and samplers.

We put push constants first (if any) in the table, followed by bind group 0
resources, followed by other bind groups. The vertex buffers are bound at the very
end of the VS buffer table.

!*/

mod adapter;
mod command;
mod conv;
mod device;
mod surface;

use std::{
    iter, ops,
    ptr::NonNull,
    sync::{atomic, Arc},
    thread,
};

use arrayvec::ArrayVec;
use foreign_types::ForeignTypeRef as _;
use parking_lot::Mutex;

#[derive(Clone)]
pub struct Api;

type ResourceIndex = u32;

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
    managed_metal_layer_delegate: surface::HalManagedMetalLayerDelegate,
}

impl Instance {
    pub fn create_surface_from_layer(&self, layer: &mtl::MetalLayerRef) -> Surface {
        unsafe { Surface::from_layer(layer) }
    }
}

impl crate::Instance<Api> for Instance {
    unsafe fn init(_desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        //TODO: enable `METAL_DEVICE_WRAPPER_TYPE` environment based on the flags?
        Ok(Instance {
            managed_metal_layer_delegate: surface::HalManagedMetalLayerDelegate::new(),
        })
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        match window_handle {
            #[cfg(target_os = "ios")]
            raw_window_handle::RawWindowHandle::UiKit(handle) => {
                let _ = &self.managed_metal_layer_delegate;
                Ok(Surface::from_view(handle.ui_view, None))
            }
            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::AppKit(handle) => Ok(Surface::from_view(
                handle.ns_view,
                Some(&self.managed_metal_layer_delegate),
            )),
            _ => Err(crate::InstanceError),
        }
    }

    unsafe fn destroy_surface(&self, surface: Surface) {
        surface.dispose();
    }

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
                        device_type: shared.private_caps.device_type(),
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

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct PrivateCapabilities {
    family_check: bool,
    msl_version: mtl::MTLLanguageVersion,
    fragment_rw_storage: bool,
    read_write_texture_tier: mtl::MTLReadWriteTextureTier,
    msaa_desktop: bool,
    msaa_apple3: bool,
    msaa_apple7: bool,
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
    format_astc_hdr: bool,
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
    max_vertex_buffers: ResourceIndex,
    max_textures_per_stage: ResourceIndex,
    max_samplers_per_stage: ResourceIndex,
    buffer_alignment: u64,
    max_buffer_size: u64,
    max_texture_size: u64,
    max_texture_3d_size: u64,
    max_texture_layers: u64,
    max_fragment_input_components: u64,
    max_color_render_targets: u8,
    max_varying_components: u32,
    max_threads_per_group: u32,
    max_total_threadgroup_memory: u32,
    sample_count_mask: u8,
    supports_debug_markers: bool,
    supports_binary_archives: bool,
    supports_capture_manager: bool,
    can_set_maximum_drawables_count: bool,
    can_set_display_sync: bool,
    can_set_next_drawable_timeout: bool,
    supports_arrays_of_textures: bool,
    supports_arrays_of_textures_write: bool,
    supports_mutability: bool,
    supports_depth_clip_control: bool,
    supports_preserve_invariance: bool,
    has_unified_memory: Option<bool>,
}

#[derive(Clone, Debug)]
struct PrivateDisabilities {
    /// Near depth is not respected properly on some Intel GPUs.
    broken_viewport_near_depth: bool,
    /// Multi-target clears don't appear to work properly on Intel GPUs.
    #[allow(dead_code)]
    broken_layered_clear_image: bool,
}

#[derive(Debug, Default)]
struct Settings {
    retain_command_buffer_references: bool,
}

struct AdapterShared {
    device: Mutex<mtl::Device>,
    disabilities: PrivateDisabilities,
    private_caps: PrivateCapabilities,
    settings: Settings,
}

unsafe impl Send for AdapterShared {}
unsafe impl Sync for AdapterShared {}

impl AdapterShared {
    fn new(device: mtl::Device) -> Self {
        let private_caps = PrivateCapabilities::new(&device);
        log::debug!("{:#?}", private_caps);

        Self {
            disabilities: PrivateDisabilities::new(&device),
            private_caps,
            device: Mutex::new(device),
            settings: Settings::default(),
        }
    }
}

pub struct Adapter {
    shared: Arc<AdapterShared>,
}

pub struct Queue {
    raw: Arc<Mutex<mtl::CommandQueue>>,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

pub struct Device {
    shared: Arc<AdapterShared>,
    features: wgt::Features,
}

pub struct Surface {
    view: Option<NonNull<objc::runtime::Object>>,
    render_layer: Mutex<mtl::MetalLayer>,
    raw_swapchain_format: mtl::MTLPixelFormat,
    extent: wgt::Extent3d,
    main_thread_id: thread::ThreadId,
    // Useful for UI-intensive applications that are sensitive to
    // window resizing.
    pub present_with_transaction: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug)]
pub struct SurfaceTexture {
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
    unsafe fn submit(
        &mut self,
        command_buffers: &[&CommandBuffer],
        signal_fence: Option<(&mut Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        objc::rc::autoreleasepool(|| {
            let extra_command_buffer = match signal_fence {
                Some((fence, value)) => {
                    let completed_value = Arc::clone(&fence.completed_value);
                    let block = block::ConcreteBlock::new(move |_cmd_buf| {
                        completed_value.store(value, atomic::Ordering::Release);
                    })
                    .copy();

                    let raw = match command_buffers.last() {
                        Some(&cmd_buf) => cmd_buf.raw.to_owned(),
                        None => {
                            let queue = self.raw.lock();
                            queue
                                .new_command_buffer_with_unretained_references()
                                .to_owned()
                        }
                    };
                    raw.set_label("(wgpu internal) Signal");
                    raw.add_completed_handler(&block);

                    fence.maintain();
                    fence.pending_command_buffers.push((value, raw.to_owned()));
                    // only return an extra one if it's extra
                    match command_buffers.last() {
                        Some(_) => None,
                        None => Some(raw),
                    }
                }
                None => None,
            };

            for cmd_buffer in command_buffers {
                cmd_buffer.raw.commit();
            }

            if let Some(raw) = extra_command_buffer {
                raw.commit();
            }
        });
        Ok(())
    }
    unsafe fn present(
        &mut self,
        _surface: &mut Surface,
        texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let queue = &self.raw.lock();
        objc::rc::autoreleasepool(|| {
            let command_buffer = queue.new_command_buffer();
            command_buffer.set_label("(wgpu internal) Present");

            // https://developer.apple.com/documentation/quartzcore/cametallayer/1478157-presentswithtransaction?language=objc
            if !texture.present_with_transaction {
                command_buffer.present_drawable(&texture.drawable);
            }

            command_buffer.commit();

            if texture.present_with_transaction {
                command_buffer.wait_until_scheduled();
                texture.drawable.present();
            }
        });
        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        // TODO: This is hard, see https://github.com/gpuweb/gpuweb/issues/1325
        1.0
    }
}

#[derive(Debug)]
pub struct Buffer {
    raw: mtl::Buffer,
    size: wgt::BufferAddress,
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
    copy_size: crate::CopyExtent,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

#[derive(Debug)]
pub struct TextureView {
    raw: mtl::Texture,
    aspects: crate::FormatAspects,
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

#[derive(Debug)]
pub struct BindGroupLayout {
    /// Sorted list of BGL entries.
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PushConstantsInfo {
    count: u32,
    buffer_index: ResourceIndex,
}

#[derive(Debug)]
pub struct PipelineLayout {
    naga_options: naga::back::msl::Options,
    bind_group_infos: ArrayVec<BindGroupLayoutInfo, { crate::MAX_BIND_GROUPS }>,
    push_constants_infos: MultiStageData<Option<PushConstantsInfo>>,
    total_counters: MultiStageResourceCounters,
    total_push_constants: u32,
}

trait AsNative {
    type Native;
    fn from(native: &Self::Native) -> Self;
    fn as_native(&self) -> &Self::Native;
}

type BufferPtr = NonNull<mtl::MTLBuffer>;
type TexturePtr = NonNull<mtl::MTLTexture>;
type SamplerPtr = NonNull<mtl::MTLSamplerState>;

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
    dynamic_index: Option<u32>,
    binding_size: Option<wgt::BufferSize>,
    binding_location: u32,
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

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
}

#[derive(Debug, Default)]
struct PipelineStageInfo {
    push_constants: Option<PushConstantsInfo>,
    sizes_slot: Option<naga::back::msl::Slot>,
    sized_bindings: Vec<naga::ResourceBinding>,
}

impl PipelineStageInfo {
    fn clear(&mut self) {
        self.push_constants = None;
        self.sizes_slot = None;
        self.sized_bindings.clear();
    }

    fn assign_from(&mut self, other: &Self) {
        self.push_constants = other.push_constants;
        self.sizes_slot = other.sizes_slot;
        self.sized_bindings.clear();
        self.sized_bindings.extend_from_slice(&other.sized_bindings);
    }
}

pub struct RenderPipeline {
    raw: mtl::RenderPipelineState,
    #[allow(dead_code)]
    vs_lib: mtl::Library,
    #[allow(dead_code)]
    fs_lib: Option<mtl::Library>,
    vs_info: PipelineStageInfo,
    fs_info: PipelineStageInfo,
    raw_primitive_type: mtl::MTLPrimitiveType,
    raw_triangle_fill_mode: mtl::MTLTriangleFillMode,
    raw_front_winding: mtl::MTLWinding,
    raw_cull_mode: mtl::MTLCullMode,
    raw_depth_clip_mode: Option<mtl::MTLDepthClipMode>,
    depth_stencil: Option<(mtl::DepthStencilState, wgt::DepthBiasState)>,
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

pub struct ComputePipeline {
    raw: mtl::ComputePipelineState,
    #[allow(dead_code)]
    cs_lib: mtl::Library,
    cs_info: PipelineStageInfo,
    work_group_size: mtl::MTLSize,
    work_group_memory_sizes: Vec<u32>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

#[derive(Debug)]
pub struct QuerySet {
    raw_buffer: mtl::Buffer,
    ty: wgt::QueryType,
}

unsafe impl Send for QuerySet {}
unsafe impl Sync for QuerySet {}

#[derive(Debug)]
pub struct Fence {
    completed_value: Arc<atomic::AtomicU64>,
    /// The pending fence values have to be ascending.
    pending_command_buffers: Vec<(crate::FenceValue, mtl::CommandBuffer)>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    fn get_latest(&self) -> crate::FenceValue {
        let mut max_value = self.completed_value.load(atomic::Ordering::Acquire);
        for &(value, ref cmd_buf) in self.pending_command_buffers.iter() {
            if cmd_buf.status() == mtl::MTLCommandBufferStatus::Completed {
                max_value = value;
            }
        }
        max_value
    }

    fn maintain(&mut self) {
        let latest = self.get_latest();
        self.pending_command_buffers
            .retain(|&(value, _)| value > latest);
    }
}

struct IndexState {
    buffer_ptr: BufferPtr,
    offset: wgt::BufferAddress,
    stride: wgt::BufferAddress,
    raw_type: mtl::MTLIndexType,
}

#[derive(Default)]
struct Temp {
    binding_sizes: Vec<u32>,
}

struct CommandState {
    blit: Option<mtl::BlitCommandEncoder>,
    render: Option<mtl::RenderCommandEncoder>,
    compute: Option<mtl::ComputeCommandEncoder>,
    raw_primitive_type: mtl::MTLPrimitiveType,
    index: Option<IndexState>,
    raw_wg_size: mtl::MTLSize,
    stage_infos: MultiStageData<PipelineStageInfo>,
    storage_buffer_length_map: fxhash::FxHashMap<naga::ResourceBinding, wgt::BufferSize>,
    work_group_memory_sizes: Vec<u32>,
    push_constants: Vec<u32>,
}

pub struct CommandEncoder {
    shared: Arc<AdapterShared>,
    raw_queue: Arc<Mutex<mtl::CommandQueue>>,
    raw_cmd_buf: Option<mtl::CommandBuffer>,
    state: CommandState,
    temp: Temp,
}

unsafe impl Send for CommandEncoder {}
unsafe impl Sync for CommandEncoder {}

pub struct CommandBuffer {
    raw: mtl::CommandBuffer,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}
