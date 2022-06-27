/*!
# DirectX12 API internals.

Generally the mapping is straightforwad.

## Resource transitions

D3D12 API matches WebGPU internal states very well. The only
caveat here is issuing a special UAV barrier whenever both source
and destination states match, and they are for storage sync.

## Memory

For now, all resources are created with "committed" memory.

## Resource binding

See ['Device::create_pipeline_layout`] documentation for the structure
of the root signature corresponding to WebGPU pipeline layout.

Binding groups is mostly straightforward, with one big caveat:
all bindings have to be reset whenever the pipeline layout changes.
This is the rule of D3D12, and we can do nothing to help it.

We detect this change at both [`crate::CommandEncoder::set_bind_group`]
and [`crate::CommandEncoder::set_render_pipeline`] with
[`crate::CommandEncoder::set_compute_pipeline`].

For this reason, in order avoid repeating the binding code,
we are binding everything in [`CommandEncoder::update_root_elements`].
When the pipeline layout is changed, we reset all bindings.
Otherwise, we pass a range corresponding only to the current bind group.

!*/

mod adapter;
mod command;
mod conv;
mod descriptor;
mod device;
mod instance;
mod view;

use crate::auxil::{self, dxgi::result::HResult as _};

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use std::{ffi, mem, num::NonZeroU32, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_4, dxgitype, windef, winerror},
    um::{d3d12, dcomp, synchapi, winbase, winnt},
    Interface as _,
};

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
    type SurfaceTexture = Texture;
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

// Limited by D3D12's root signature size of 64. Each element takes 1 or 2 entries.
const MAX_ROOT_ELEMENTS: usize = 64;
const ZERO_BUFFER_SIZE: wgt::BufferAddress = 256 << 10;

pub struct Instance {
    factory: native::DxgiFactory,
    library: Arc<native::D3D12Lib>,
    _lib_dxgi: native::DxgiLib,
    flags: crate::InstanceFlags,
}

impl Instance {
    pub unsafe fn create_surface_from_visual(
        &self,
        visual: *mut dcomp::IDCompositionVisual,
    ) -> Surface {
        Surface {
            factory: self.factory,
            target: SurfaceTarget::Visual(native::WeakPtr::from_raw(visual)),
            swap_chain: None,
        }
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

struct SwapChain {
    raw: native::WeakPtr<dxgi1_4::IDXGISwapChain3>,
    // need to associate raw image pointers with the swapchain so they can be properly released
    // when the swapchain is destroyed
    resources: Vec<native::Resource>,
    waitable: winnt::HANDLE,
    acquired_count: usize,
    present_mode: wgt::PresentMode,
    format: wgt::TextureFormat,
    size: wgt::Extent3d,
}

enum SurfaceTarget {
    WndHandle(windef::HWND),
    Visual(native::WeakPtr<dcomp::IDCompositionVisual>),
}

pub struct Surface {
    factory: native::DxgiFactory,
    target: SurfaceTarget,
    swap_chain: Option<SwapChain>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug, Clone, Copy)]
enum MemoryArchitecture {
    Unified {
        #[allow(unused)]
        cache_coherent: bool,
    },
    NonUnified,
}

#[derive(Debug, Clone, Copy)]
struct PrivateCapabilities {
    instance_flags: crate::InstanceFlags,
    #[allow(unused)]
    heterogeneous_resource_heaps: bool,
    memory_architecture: MemoryArchitecture,
    heap_create_not_zeroed: bool,
}

#[derive(Default)]
struct Workarounds {
    // On WARP, temporary CPU descriptors are still used by the runtime
    // after we call `CopyDescriptors`.
    avoid_cpu_descriptor_overwrites: bool,
}

pub struct Adapter {
    raw: native::DxgiAdapter,
    device: native::Device,
    library: Arc<native::D3D12Lib>,
    private_caps: PrivateCapabilities,
    //Note: this isn't used right now, but we'll need it later.
    #[allow(unused)]
    workarounds: Workarounds,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

/// Helper structure for waiting for GPU.
struct Idler {
    fence: native::Fence,
    event: native::Event,
}

impl Idler {
    unsafe fn destroy(self) {
        self.fence.destroy();
    }
}

struct CommandSignatures {
    draw: native::CommandSignature,
    draw_indexed: native::CommandSignature,
    dispatch: native::CommandSignature,
}

impl CommandSignatures {
    unsafe fn destroy(&self) {
        self.draw.destroy();
        self.draw_indexed.destroy();
        self.dispatch.destroy();
    }
}

struct DeviceShared {
    zero_buffer: native::Resource,
    cmd_signatures: CommandSignatures,
    heap_views: descriptor::GeneralHeap,
    heap_samplers: descriptor::GeneralHeap,
}

impl DeviceShared {
    unsafe fn destroy(&self) {
        self.zero_buffer.destroy();
        self.cmd_signatures.destroy();
        self.heap_views.raw.destroy();
        self.heap_samplers.raw.destroy();
    }
}

pub struct Device {
    raw: native::Device,
    present_queue: native::CommandQueue,
    idler: Idler,
    private_caps: PrivateCapabilities,
    shared: Arc<DeviceShared>,
    // CPU only pools
    rtv_pool: Mutex<descriptor::CpuPool>,
    dsv_pool: Mutex<descriptor::CpuPool>,
    srv_uav_pool: Mutex<descriptor::CpuPool>,
    sampler_pool: Mutex<descriptor::CpuPool>,
    // library
    library: Arc<native::D3D12Lib>,
    #[cfg(feature = "renderdoc")]
    render_doc: crate::auxil::renderdoc::RenderDoc,
    null_rtv_handle: descriptor::Handle,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: native::CommandQueue,
    temp_lists: Vec<native::CommandList>,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

#[derive(Default)]
struct Temp {
    marker: Vec<u16>,
    barriers: Vec<d3d12::D3D12_RESOURCE_BARRIER>,
}

impl Temp {
    fn clear(&mut self) {
        self.marker.clear();
        self.barriers.clear();
    }
}

struct PassResolve {
    src: (native::Resource, u32),
    dst: (native::Resource, u32),
    format: native::Format,
}

#[derive(Clone, Copy)]
enum RootElement {
    Empty,
    SpecialConstantBuffer {
        base_vertex: i32,
        base_instance: u32,
        other: u32,
    },
    /// Descriptor table.
    Table(native::GpuDescriptor),
    /// Descriptor for a buffer that has dynamic offset.
    DynamicOffsetBuffer {
        kind: BufferViewKind,
        address: native::GpuAddress,
    },
}

#[derive(Clone, Copy)]
enum PassKind {
    Render,
    Compute,
    Transfer,
}

struct PassState {
    has_label: bool,
    resolves: ArrayVec<PassResolve, { crate::MAX_COLOR_ATTACHMENTS }>,
    layout: PipelineLayoutShared,
    root_elements: [RootElement; MAX_ROOT_ELEMENTS],
    dirty_root_elements: u64,
    vertex_buffers: [d3d12::D3D12_VERTEX_BUFFER_VIEW; crate::MAX_VERTEX_BUFFERS],
    dirty_vertex_buffers: usize,
    kind: PassKind,
}

#[test]
fn test_dirty_mask() {
    assert_eq!(MAX_ROOT_ELEMENTS, std::mem::size_of::<u64>() * 8);
}

impl PassState {
    fn new() -> Self {
        PassState {
            has_label: false,
            resolves: ArrayVec::new(),
            layout: PipelineLayoutShared {
                signature: native::RootSignature::null(),
                total_root_elements: 0,
                special_constants_root_index: None,
            },
            root_elements: [RootElement::Empty; MAX_ROOT_ELEMENTS],
            dirty_root_elements: 0,
            vertex_buffers: [unsafe { mem::zeroed() }; crate::MAX_VERTEX_BUFFERS],
            dirty_vertex_buffers: 0,
            kind: PassKind::Transfer,
        }
    }

    fn clear(&mut self) {
        // careful about heap allocations!
        *self = Self::new();
    }
}

pub struct CommandEncoder {
    allocator: native::CommandAllocator,
    device: native::Device,
    shared: Arc<DeviceShared>,
    null_rtv_handle: descriptor::Handle,
    list: Option<native::GraphicsCommandList>,
    free_lists: Vec<native::GraphicsCommandList>,
    pass: PassState,
    temp: Temp,
}

unsafe impl Send for CommandEncoder {}
unsafe impl Sync for CommandEncoder {}

pub struct CommandBuffer {
    raw: native::GraphicsCommandList,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

#[derive(Debug)]
pub struct Buffer {
    resource: native::Resource,
    size: wgt::BufferAddress,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl crate::BufferBinding<'_, Api> {
    fn resolve_size(&self) -> wgt::BufferAddress {
        match self.size {
            Some(size) => size.get(),
            None => self.buffer.size - self.offset,
        }
    }

    fn resolve_address(&self) -> wgt::BufferAddress {
        self.buffer.resource.gpu_virtual_address() + self.offset
    }
}

#[derive(Debug)]
pub struct Texture {
    resource: native::Resource,
    format: wgt::TextureFormat,
    dimension: wgt::TextureDimension,
    size: wgt::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Texture {
    fn array_layer_count(&self) -> u32 {
        match self.dimension {
            wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
                self.size.depth_or_array_layers
            }
            wgt::TextureDimension::D3 => 1,
        }
    }

    fn calc_subresource(&self, mip_level: u32, array_layer: u32, plane: u32) -> u32 {
        mip_level + (array_layer + plane * self.array_layer_count()) * self.mip_level_count
    }

    fn calc_subresource_for_copy(&self, base: &crate::TextureCopyBase) -> u32 {
        self.calc_subresource(base.mip_level, base.array_layer, 0)
    }
}

#[derive(Debug)]
pub struct TextureView {
    raw_format: native::Format,
    format_aspects: crate::FormatAspects, // May explicitly ignore stencil aspect of raw_format!
    target_base: (native::Resource, u32),
    handle_srv: Option<descriptor::Handle>,
    handle_uav: Option<descriptor::Handle>,
    handle_rtv: Option<descriptor::Handle>,
    handle_dsv_ro: Option<descriptor::Handle>,
    handle_dsv_rw: Option<descriptor::Handle>,
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

#[derive(Debug)]
pub struct Sampler {
    handle: descriptor::Handle,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

#[derive(Debug)]
pub struct QuerySet {
    raw: native::QueryHeap,
    raw_ty: d3d12::D3D12_QUERY_TYPE,
}

unsafe impl Send for QuerySet {}
unsafe impl Sync for QuerySet {}

#[derive(Debug)]
pub struct Fence {
    raw: native::Fence,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

pub struct BindGroupLayout {
    /// Sorted list of entries.
    entries: Vec<wgt::BindGroupLayoutEntry>,
    cpu_heap_views: Option<descriptor::CpuHeap>,
    cpu_heap_samplers: Option<descriptor::CpuHeap>,
    copy_counts: Vec<u32>, // all 1's
}

#[derive(Clone, Copy)]
enum BufferViewKind {
    Constant,
    ShaderResource,
    UnorderedAccess,
}

#[derive(Debug)]
pub struct BindGroup {
    handle_views: Option<descriptor::DualHandle>,
    handle_samplers: Option<descriptor::DualHandle>,
    dynamic_buffers: Vec<native::GpuAddress>,
}

bitflags::bitflags! {
    struct TableTypes: u8 {
        const SRV_CBV_UAV = 1 << 0;
        const SAMPLERS = 1 << 1;
    }
}

// Element (also known as parameter) index into the root signature.
type RootIndex = u32;

struct BindGroupInfo {
    base_root_index: RootIndex,
    tables: TableTypes,
    dynamic_buffers: Vec<BufferViewKind>,
}

#[derive(Clone)]
struct PipelineLayoutShared {
    signature: native::RootSignature,
    total_root_elements: RootIndex,
    special_constants_root_index: Option<RootIndex>,
}

unsafe impl Send for PipelineLayoutShared {}
unsafe impl Sync for PipelineLayoutShared {}

pub struct PipelineLayout {
    shared: PipelineLayoutShared,
    // Storing for each associated bind group, which tables we created
    // in the root signature. This is required for binding descriptor sets.
    bind_group_infos: ArrayVec<BindGroupInfo, { crate::MAX_BIND_GROUPS }>,
    naga_options: naga::back::hlsl::Options,
}

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
    raw_name: Option<ffi::CString>,
}

pub struct RenderPipeline {
    raw: native::PipelineState,
    layout: PipelineLayoutShared,
    topology: d3d12::D3D12_PRIMITIVE_TOPOLOGY,
    vertex_strides: [Option<NonZeroU32>; crate::MAX_VERTEX_BUFFERS],
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

pub struct ComputePipeline {
    raw: native::PipelineState,
    layout: PipelineLayoutShared,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl SwapChain {
    unsafe fn release_resources(self) -> native::WeakPtr<dxgi1_4::IDXGISwapChain3> {
        for resource in self.resources {
            resource.destroy();
        }
        self.raw
    }

    unsafe fn wait(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<bool, crate::SurfaceError> {
        let timeout_ms = match timeout {
            Some(duration) => duration.as_millis() as u32,
            None => winbase::INFINITE,
        };
        match synchapi::WaitForSingleObject(self.waitable, timeout_ms) {
            winbase::WAIT_ABANDONED | winbase::WAIT_FAILED => Err(crate::SurfaceError::Lost),
            winbase::WAIT_OBJECT_0 => Ok(true),
            winerror::WAIT_TIMEOUT => Ok(false),
            other => {
                log::error!("Unexpected wait status: 0x{:x}", other);
                Err(crate::SurfaceError::Lost)
            }
        }
    }
}

impl crate::Surface<Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let mut flags = dxgi::DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        match config.present_mode {
            wgt::PresentMode::Immediate => {
                flags |= dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
            }
            _ => {}
        }

        let non_srgb_format = auxil::dxgi::conv::map_texture_format_nosrgb(config.format);

        let swap_chain = match self.swap_chain.take() {
            //Note: this path doesn't properly re-initialize all of the things
            Some(sc) => {
                // can't have image resources in flight used by GPU
                let _ = device.wait_idle();

                let raw = sc.release_resources();
                let result = raw.ResizeBuffers(
                    config.swap_chain_size,
                    config.extent.width,
                    config.extent.height,
                    non_srgb_format,
                    flags,
                );
                if let Err(err) = result.into_result() {
                    log::error!("ResizeBuffers failed: {}", err);
                    return Err(crate::SurfaceError::Other("window is in use"));
                }
                raw
            }
            None => {
                let desc = native::SwapchainDesc {
                    alpha_mode: auxil::dxgi::conv::map_acomposite_alpha_mode(
                        config.composite_alpha_mode,
                    ),
                    width: config.extent.width,
                    height: config.extent.height,
                    format: non_srgb_format,
                    stereo: false,
                    sample: native::SampleDesc {
                        count: 1,
                        quality: 0,
                    },
                    buffer_usage: dxgitype::DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    buffer_count: config.swap_chain_size,
                    scaling: native::Scaling::Stretch,
                    swap_effect: native::SwapEffect::FlipDiscard,
                    flags,
                };
                let swap_chain1 = match self.target {
                    SurfaceTarget::Visual(_) => {
                        profiling::scope!("IDXGIFactory4::CreateSwapChainForComposition");
                        self.factory
                            .unwrap_factory2()
                            .create_swapchain_for_composition(
                                device.present_queue.as_mut_ptr() as *mut _,
                                &desc,
                            )
                            .into_result()
                    }
                    SurfaceTarget::WndHandle(hwnd) => {
                        profiling::scope!("IDXGIFactory4::CreateSwapChainForHwnd");
                        self.factory
                            .as_factory2()
                            .unwrap()
                            .create_swapchain_for_hwnd(
                                device.present_queue.as_mut_ptr() as *mut _,
                                hwnd,
                                &desc,
                            )
                            .into_result()
                    }
                };

                let swap_chain1 = match swap_chain1 {
                    Ok(s) => s,
                    Err(err) => {
                        log::error!("SwapChain creation error: {}", err);
                        return Err(crate::SurfaceError::Other("swap chain creation"));
                    }
                };

                match self.target {
                    SurfaceTarget::WndHandle(_) => {}
                    SurfaceTarget::Visual(visual) => {
                        if let Err(err) = visual.SetContent(swap_chain1.as_unknown()).into_result()
                        {
                            log::error!("Unable to SetContent: {}", err);
                            return Err(crate::SurfaceError::Other(
                                "IDCompositionVisual::SetContent",
                            ));
                        }
                    }
                }

                match swap_chain1.cast::<dxgi1_4::IDXGISwapChain3>().into_result() {
                    Ok(swap_chain3) => {
                        swap_chain1.destroy();
                        swap_chain3
                    }
                    Err(err) => {
                        log::error!("Unable to cast swap chain: {}", err);
                        return Err(crate::SurfaceError::Other("swap chain cast to 3"));
                    }
                }
            }
        };

        match self.target {
            SurfaceTarget::WndHandle(wnd_handle) => {
                // Disable automatic Alt+Enter handling by DXGI.
                const DXGI_MWA_NO_WINDOW_CHANGES: u32 = 1;
                const DXGI_MWA_NO_ALT_ENTER: u32 = 2;
                self.factory.MakeWindowAssociation(
                    wnd_handle,
                    DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER,
                );
            }
            SurfaceTarget::Visual(_) => {}
        }

        swap_chain.SetMaximumFrameLatency(config.swap_chain_size);
        let waitable = swap_chain.GetFrameLatencyWaitableObject();

        let mut resources = vec![native::Resource::null(); config.swap_chain_size as usize];
        for (i, res) in resources.iter_mut().enumerate() {
            swap_chain.GetBuffer(i as _, &d3d12::ID3D12Resource::uuidof(), res.mut_void());
        }

        self.swap_chain = Some(SwapChain {
            raw: swap_chain,
            resources,
            waitable,
            acquired_count: 0,
            present_mode: config.present_mode,
            format: config.format,
            size: config.extent,
        });

        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &Device) {
        if let Some(mut sc) = self.swap_chain.take() {
            let _ = sc.wait(None);
            //TODO: this shouldn't be needed,
            // but it complains that the queue is still used otherwise
            let _ = device.wait_idle();
            let raw = sc.release_resources();
            raw.destroy();
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        let sc = self.swap_chain.as_mut().unwrap();

        sc.wait(timeout)?;

        let base_index = sc.raw.GetCurrentBackBufferIndex() as usize;
        let index = (base_index + sc.acquired_count) % sc.resources.len();
        sc.acquired_count += 1;

        let texture = Texture {
            resource: sc.resources[index],
            format: sc.format,
            dimension: wgt::TextureDimension::D2,
            size: sc.size,
            mip_level_count: 1,
            sample_count: 1,
        };
        Ok(Some(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal: false,
        }))
    }
    unsafe fn discard_texture(&mut self, _texture: Texture) {
        let sc = self.swap_chain.as_mut().unwrap();
        sc.acquired_count -= 1;
    }
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&CommandBuffer],
        signal_fence: Option<(&mut Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        self.temp_lists.clear();
        for cmd_buf in command_buffers {
            self.temp_lists.push(cmd_buf.raw.as_list());
        }

        {
            profiling::scope!("ID3D12CommandQueue::ExecuteCommandLists");
            self.raw.execute_command_lists(&self.temp_lists);
        }

        if let Some((fence, value)) = signal_fence {
            self.raw
                .signal(fence.raw, value)
                .into_device_result("Signal fence")?;
        }
        Ok(())
    }
    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        _texture: Texture,
    ) -> Result<(), crate::SurfaceError> {
        let sc = surface.swap_chain.as_mut().unwrap();
        sc.acquired_count -= 1;

        let (interval, flags) = match sc.present_mode {
            wgt::PresentMode::Immediate => (0, dxgi::DXGI_PRESENT_ALLOW_TEARING),
            wgt::PresentMode::Fifo => (1, 0),
            wgt::PresentMode::Mailbox => (1, 0),
        };

        profiling::scope!("IDXGISwapchain3::Present");
        sc.raw.Present(interval, flags);

        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        let mut frequency = 0u64;
        self.raw.GetTimestampFrequency(&mut frequency);
        (1_000_000_000.0 / frequency as f64) as f32
    }
}
