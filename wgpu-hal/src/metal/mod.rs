#![allow(unused_variables)]

mod adapter;
mod command;
mod conv;
mod device;
mod surface;

use std::{ops::Range, ptr::NonNull, sync::Arc, thread};

use parking_lot::Mutex;

#[derive(Clone)]
pub struct Api;
pub struct Context;
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

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &Resource, dst: &Resource, regions: T)
    where
        T: Iterator<Item = crate::BufferCopy>,
    {
    }

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &Resource,
        src_usage: crate::TextureUse,
        dst: &Resource,
        regions: T,
    ) {
    }

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Resource, dst: &Resource, regions: T) {}

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Resource,
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

    unsafe fn begin_compute_pass(&mut self) {}
    unsafe fn end_compute_pass(&mut self) {}

    unsafe fn set_compute_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_indirect(&mut self, buffer: &Resource, offset: wgt::BufferAddress) {}
}
