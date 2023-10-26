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
we are binding everything in `CommandEncoder::update_root_elements`.
When the pipeline layout is changed, we reset all bindings.
Otherwise, we pass a range corresponding only to the current bind group.

!*/

mod adapter;
mod command;
mod conv;
mod descriptor;
mod device;
mod instance;
mod shader_compilation;
mod suballocation;
mod types;
mod view;

use crate::auxil::{self, dxgi::result::HResult as _};

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use std::{ffi, fmt, mem, num::NonZeroU32, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_4, dxgitype, windef, winerror},
    um::{d3d12 as d3d12_ty, dcomp, synchapi, winbase, winnt},
    Interface as _,
};

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
    factory: d3d12::DxgiFactory,
    factory_media: Option<d3d12::FactoryMedia>,
    library: Arc<d3d12::D3D12Lib>,
    supports_allow_tearing: bool,
    _lib_dxgi: d3d12::DxgiLib,
    flags: wgt::InstanceFlags,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
}

impl Instance {
    pub unsafe fn create_surface_from_visual(
        &self,
        visual: *mut dcomp::IDCompositionVisual,
    ) -> Surface {
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::Visual(unsafe { d3d12::ComPtr::from_raw(visual) }),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: None,
        }
    }

    pub unsafe fn create_surface_from_surface_handle(
        &self,
        surface_handle: winnt::HANDLE,
    ) -> Surface {
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::SurfaceHandle(surface_handle),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: None,
        }
    }

    pub unsafe fn create_surface_from_swap_chain_panel(
        &self,
        swap_chain_panel: *mut types::ISwapChainPanelNative,
    ) -> Surface {
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::SwapChainPanel(unsafe {
                d3d12::ComPtr::from_raw(swap_chain_panel)
            }),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: None,
        }
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

struct SwapChain {
    raw: d3d12::ComPtr<dxgi1_4::IDXGISwapChain3>,
    // need to associate raw image pointers with the swapchain so they can be properly released
    // when the swapchain is destroyed
    resources: Vec<d3d12::Resource>,
    waitable: winnt::HANDLE,
    acquired_count: usize,
    present_mode: wgt::PresentMode,
    format: wgt::TextureFormat,
    size: wgt::Extent3d,
}

enum SurfaceTarget {
    WndHandle(windef::HWND),
    Visual(d3d12::ComPtr<dcomp::IDCompositionVisual>),
    SurfaceHandle(winnt::HANDLE),
    SwapChainPanel(d3d12::ComPtr<types::ISwapChainPanelNative>),
}

pub struct Surface {
    factory: d3d12::DxgiFactory,
    factory_media: Option<d3d12::FactoryMedia>,
    target: SurfaceTarget,
    supports_allow_tearing: bool,
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
    instance_flags: wgt::InstanceFlags,
    #[allow(unused)]
    heterogeneous_resource_heaps: bool,
    memory_architecture: MemoryArchitecture,
    #[allow(unused)] // TODO: Exists until windows-rs is standard, then it can probably be removed?
    heap_create_not_zeroed: bool,
    casting_fully_typed_format_supported: bool,
    suballocation_supported: bool,
}

#[derive(Default)]
struct Workarounds {
    // On WARP, temporary CPU descriptors are still used by the runtime
    // after we call `CopyDescriptors`.
    avoid_cpu_descriptor_overwrites: bool,
}

pub struct Adapter {
    raw: d3d12::DxgiAdapter,
    device: d3d12::Device,
    library: Arc<d3d12::D3D12Lib>,
    private_caps: PrivateCapabilities,
    presentation_timer: auxil::dxgi::time::PresentationTimer,
    //Note: this isn't used right now, but we'll need it later.
    #[allow(unused)]
    workarounds: Workarounds,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

/// Helper structure for waiting for GPU.
struct Idler {
    fence: d3d12::Fence,
    event: d3d12::Event,
}

struct CommandSignatures {
    draw: d3d12::CommandSignature,
    draw_indexed: d3d12::CommandSignature,
    dispatch: d3d12::CommandSignature,
}

struct DeviceShared {
    zero_buffer: d3d12::Resource,
    cmd_signatures: CommandSignatures,
    heap_views: descriptor::GeneralHeap,
    heap_samplers: descriptor::GeneralHeap,
}

pub struct Device {
    raw: d3d12::Device,
    present_queue: d3d12::CommandQueue,
    idler: Idler,
    private_caps: PrivateCapabilities,
    shared: Arc<DeviceShared>,
    // CPU only pools
    rtv_pool: Mutex<descriptor::CpuPool>,
    dsv_pool: Mutex<descriptor::CpuPool>,
    srv_uav_pool: Mutex<descriptor::CpuPool>,
    sampler_pool: Mutex<descriptor::CpuPool>,
    // library
    library: Arc<d3d12::D3D12Lib>,
    #[cfg(feature = "renderdoc")]
    render_doc: crate::auxil::renderdoc::RenderDoc,
    null_rtv_handle: descriptor::Handle,
    mem_allocator: Option<Mutex<suballocation::GpuAllocatorWrapper>>,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: d3d12::CommandQueue,
    temp_lists: Vec<d3d12::CommandList>,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

#[derive(Default)]
struct Temp {
    marker: Vec<u16>,
    barriers: Vec<d3d12_ty::D3D12_RESOURCE_BARRIER>,
}

impl Temp {
    fn clear(&mut self) {
        self.marker.clear();
        self.barriers.clear();
    }
}

struct PassResolve {
    src: (d3d12::Resource, u32),
    dst: (d3d12::Resource, u32),
    format: d3d12::Format,
}

#[derive(Clone, Copy)]
enum RootElement {
    Empty,
    Constant,
    SpecialConstantBuffer {
        base_vertex: i32,
        base_instance: u32,
        other: u32,
    },
    /// Descriptor table.
    Table(d3d12::GpuDescriptor),
    /// Descriptor for a buffer that has dynamic offset.
    DynamicOffsetBuffer {
        kind: BufferViewKind,
        address: d3d12::GpuAddress,
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
    constant_data: [u32; MAX_ROOT_ELEMENTS],
    dirty_root_elements: u64,
    vertex_buffers: [d3d12_ty::D3D12_VERTEX_BUFFER_VIEW; crate::MAX_VERTEX_BUFFERS],
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
                signature: d3d12::RootSignature::null(),
                total_root_elements: 0,
                special_constants_root_index: None,
                root_constant_info: None,
            },
            root_elements: [RootElement::Empty; MAX_ROOT_ELEMENTS],
            constant_data: [0; MAX_ROOT_ELEMENTS],
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
    allocator: d3d12::CommandAllocator,
    device: d3d12::Device,
    shared: Arc<DeviceShared>,
    null_rtv_handle: descriptor::Handle,
    list: Option<d3d12::GraphicsCommandList>,
    free_lists: Vec<d3d12::GraphicsCommandList>,
    pass: PassState,
    temp: Temp,

    /// If set, the end of the next render/compute pass will write a timestamp at
    /// the given pool & location.
    end_of_pass_timer_query: Option<(d3d12::QueryHeap, u32)>,
}

unsafe impl Send for CommandEncoder {}
unsafe impl Sync for CommandEncoder {}

impl fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandEncoder")
            .field("allocator", &self.allocator)
            .field("device", &self.allocator)
            .finish()
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    raw: d3d12::GraphicsCommandList,
    closed: bool,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

#[derive(Debug)]
pub struct Buffer {
    resource: d3d12::Resource,
    size: wgt::BufferAddress,
    allocation: Option<suballocation::AllocationWrapper>,
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
    resource: d3d12::Resource,
    format: wgt::TextureFormat,
    dimension: wgt::TextureDimension,
    size: wgt::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    allocation: Option<suballocation::AllocationWrapper>,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Texture {
    fn array_layer_count(&self) -> u32 {
        match self.dimension {
            wgt::TextureDimension::D1 | wgt::TextureDimension::D3 => 1,
            wgt::TextureDimension::D2 => self.size.depth_or_array_layers,
        }
    }

    /// see https://learn.microsoft.com/en-us/windows/win32/direct3d12/subresources#plane-slice
    fn calc_subresource(&self, mip_level: u32, array_layer: u32, plane: u32) -> u32 {
        mip_level + (array_layer + plane * self.array_layer_count()) * self.mip_level_count
    }

    fn calc_subresource_for_copy(&self, base: &crate::TextureCopyBase) -> u32 {
        let plane = match base.aspect {
            crate::FormatAspects::COLOR | crate::FormatAspects::DEPTH => 0,
            crate::FormatAspects::STENCIL => 1,
            _ => unreachable!(),
        };
        self.calc_subresource(base.mip_level, base.array_layer, plane)
    }
}

#[derive(Debug)]
pub struct TextureView {
    raw_format: d3d12::Format,
    aspects: crate::FormatAspects,
    target_base: (d3d12::Resource, u32),
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
    raw: d3d12::QueryHeap,
    raw_ty: d3d12_ty::D3D12_QUERY_TYPE,
}

unsafe impl Send for QuerySet {}
unsafe impl Sync for QuerySet {}

#[derive(Debug)]
pub struct Fence {
    raw: d3d12::Fence,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

#[derive(Debug)]
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
    dynamic_buffers: Vec<d3d12::GpuAddress>,
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
struct RootConstantInfo {
    root_index: RootIndex,
    range: std::ops::Range<u32>,
}

#[derive(Clone)]
struct PipelineLayoutShared {
    signature: d3d12::RootSignature,
    total_root_elements: RootIndex,
    special_constants_root_index: Option<RootIndex>,
    root_constant_info: Option<RootConstantInfo>,
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

pub(super) enum CompiledShader {
    #[allow(unused)]
    Dxc(Vec<u8>),
    Fxc(d3d12::Blob),
}

impl CompiledShader {
    fn create_native_shader(&self) -> d3d12::Shader {
        match *self {
            CompiledShader::Dxc(ref shader) => d3d12::Shader::from_raw(shader),
            CompiledShader::Fxc(ref shader) => d3d12::Shader::from_blob(shader),
        }
    }

    unsafe fn destroy(self) {}
}

pub struct RenderPipeline {
    raw: d3d12::PipelineState,
    layout: PipelineLayoutShared,
    topology: d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY,
    vertex_strides: [Option<NonZeroU32>; crate::MAX_VERTEX_BUFFERS],
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

pub struct ComputePipeline {
    raw: d3d12::PipelineState,
    layout: PipelineLayoutShared,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl SwapChain {
    unsafe fn release_resources(self) -> d3d12::ComPtr<dxgi1_4::IDXGISwapChain3> {
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
        match unsafe { synchapi::WaitForSingleObject(self.waitable, timeout_ms) } {
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
        // We always set ALLOW_TEARING on the swapchain no matter
        // what kind of swapchain we want because ResizeBuffers
        // cannot change the swapchain's ALLOW_TEARING flag.
        //
        // This does not change the behavior of the swapchain, just
        // allow present calls to use tearing.
        if self.supports_allow_tearing {
            flags |= dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }

        // While `configure`s contract ensures that no work on the GPU's main queues
        // are in flight, we still need to wait for the present queue to be idle.
        unsafe { device.wait_for_present_queue_idle() }?;

        let non_srgb_format = auxil::dxgi::conv::map_texture_format_nosrgb(config.format);

        let swap_chain = match self.swap_chain.take() {
            //Note: this path doesn't properly re-initialize all of the things
            Some(sc) => {
                let raw = unsafe { sc.release_resources() };
                let result = unsafe {
                    raw.ResizeBuffers(
                        config.swap_chain_size,
                        config.extent.width,
                        config.extent.height,
                        non_srgb_format,
                        flags,
                    )
                };
                if let Err(err) = result.into_result() {
                    log::error!("ResizeBuffers failed: {}", err);
                    return Err(crate::SurfaceError::Other("window is in use"));
                }
                raw
            }
            None => {
                let desc = d3d12::SwapchainDesc {
                    alpha_mode: auxil::dxgi::conv::map_acomposite_alpha_mode(
                        config.composite_alpha_mode,
                    ),
                    width: config.extent.width,
                    height: config.extent.height,
                    format: non_srgb_format,
                    stereo: false,
                    sample: d3d12::SampleDesc {
                        count: 1,
                        quality: 0,
                    },
                    buffer_usage: dxgitype::DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    buffer_count: config.swap_chain_size,
                    scaling: d3d12::Scaling::Stretch,
                    swap_effect: d3d12::SwapEffect::FlipDiscard,
                    flags,
                };
                let swap_chain1 = match self.target {
                    SurfaceTarget::Visual(_) | SurfaceTarget::SwapChainPanel(_) => {
                        profiling::scope!("IDXGIFactory4::CreateSwapChainForComposition");
                        self.factory
                            .unwrap_factory2()
                            .create_swapchain_for_composition(
                                device.present_queue.as_mut_ptr() as *mut _,
                                &desc,
                            )
                            .into_result()
                    }
                    SurfaceTarget::SurfaceHandle(handle) => {
                        profiling::scope!(
                            "IDXGIFactoryMedia::CreateSwapChainForCompositionSurfaceHandle"
                        );
                        self.factory_media
                            .clone()
                            .ok_or(crate::SurfaceError::Other("IDXGIFactoryMedia not found"))?
                            .create_swapchain_for_composition_surface_handle(
                                device.present_queue.as_mut_ptr() as *mut _,
                                handle,
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

                match &self.target {
                    &SurfaceTarget::WndHandle(_) | &SurfaceTarget::SurfaceHandle(_) => {}
                    &SurfaceTarget::Visual(ref visual) => {
                        if let Err(err) =
                            unsafe { visual.SetContent(swap_chain1.as_unknown()) }.into_result()
                        {
                            log::error!("Unable to SetContent: {}", err);
                            return Err(crate::SurfaceError::Other(
                                "IDCompositionVisual::SetContent",
                            ));
                        }
                    }
                    &SurfaceTarget::SwapChainPanel(ref swap_chain_panel) => {
                        if let Err(err) =
                            unsafe { swap_chain_panel.SetSwapChain(swap_chain1.as_ptr()) }
                                .into_result()
                        {
                            log::error!("Unable to SetSwapChain: {}", err);
                            return Err(crate::SurfaceError::Other(
                                "ISwapChainPanelNative::SetSwapChain",
                            ));
                        }
                    }
                }

                match unsafe { swap_chain1.cast::<dxgi1_4::IDXGISwapChain3>() }.into_result() {
                    Ok(swap_chain3) => swap_chain3,
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
                unsafe {
                    self.factory.MakeWindowAssociation(
                        wnd_handle,
                        DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER,
                    )
                };
            }
            SurfaceTarget::Visual(_)
            | SurfaceTarget::SurfaceHandle(_)
            | SurfaceTarget::SwapChainPanel(_) => {}
        }

        unsafe { swap_chain.SetMaximumFrameLatency(config.swap_chain_size) };
        let waitable = unsafe { swap_chain.GetFrameLatencyWaitableObject() };

        let mut resources = Vec::with_capacity(config.swap_chain_size as usize);
        for i in 0..config.swap_chain_size {
            let mut resource = d3d12::Resource::null();
            unsafe {
                swap_chain.GetBuffer(i, &d3d12_ty::ID3D12Resource::uuidof(), resource.mut_void())
            };
            resources.push(resource);
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
        if let Some(sc) = self.swap_chain.take() {
            unsafe {
                // While `unconfigure`s contract ensures that no work on the GPU's main queues
                // are in flight, we still need to wait for the present queue to be idle.

                // The major failure mode of this function is device loss,
                // which if we have lost the device, we should just continue
                // cleaning up, without error.
                let _ = device.wait_for_present_queue_idle();

                let _raw = sc.release_resources();
            }
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        let sc = self.swap_chain.as_mut().unwrap();

        unsafe { sc.wait(timeout) }?;

        let base_index = unsafe { sc.raw.GetCurrentBackBufferIndex() } as usize;
        let index = (base_index + sc.acquired_count) % sc.resources.len();
        sc.acquired_count += 1;

        let texture = Texture {
            resource: sc.resources[index].clone(),
            format: sc.format,
            dimension: wgt::TextureDimension::D2,
            size: sc.size,
            mip_level_count: 1,
            sample_count: 1,
            allocation: None,
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
                .signal(&fence.raw, value)
                .into_device_result("Signal fence")?;
        }

        // Note the lack of synchronization here between the main Direct queue
        // and the dedicated presentation queue. This is automatically handled
        // by the D3D runtime by detecting uses of resources derived from the
        // swapchain. This automatic detection is why you cannot use a swapchain
        // as an UAV in D3D12.

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
            // We only allow immediate if ALLOW_TEARING is valid.
            wgt::PresentMode::Immediate => (0, dxgi::DXGI_PRESENT_ALLOW_TEARING),
            wgt::PresentMode::Mailbox => (0, 0),
            wgt::PresentMode::Fifo => (1, 0),
            m => unreachable!("Cannot make surface with present mode {m:?}"),
        };

        profiling::scope!("IDXGISwapchain3::Present");
        unsafe { sc.raw.Present(interval, flags) };

        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        let mut frequency = 0u64;
        unsafe { self.raw.GetTimestampFrequency(&mut frequency) };
        (1_000_000_000.0 / frequency as f64) as f32
    }
}
