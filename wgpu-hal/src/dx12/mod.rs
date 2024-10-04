/*!
# DirectX12 API internals.

Generally the mapping is straightforward.

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

use std::{ffi, fmt, mem, num::NonZeroU32, ops::Deref, sync::Arc};

use arrayvec::ArrayVec;
use parking_lot::{Mutex, RwLock};
use windows::{
    core::{Free, Interface},
    Win32::{
        Foundation,
        Graphics::{Direct3D, Direct3D12, DirectComposition, Dxgi},
        System::Threading,
    },
};

use crate::auxil::{
    self,
    dxgi::{
        factory::{DxgiAdapter, DxgiFactory},
        result::HResult,
    },
};

#[derive(Debug)]
struct DynLib {
    inner: libloading::Library,
}

impl DynLib {
    unsafe fn new<P>(filename: P) -> Result<Self, libloading::Error>
    where
        P: AsRef<ffi::OsStr>,
    {
        unsafe { libloading::Library::new(filename) }.map(|inner| Self { inner })
    }

    unsafe fn get<T>(
        &self,
        symbol: &[u8],
    ) -> Result<libloading::Symbol<'_, T>, crate::DeviceError> {
        unsafe { self.inner.get(symbol) }.map_err(|e| match e {
            libloading::Error::GetProcAddress { .. } | libloading::Error::GetProcAddressUnknown => {
                crate::DeviceError::Unexpected
            }
            libloading::Error::IncompatibleSize
            | libloading::Error::CreateCString { .. }
            | libloading::Error::CreateCStringWithTrailing { .. } => crate::hal_internal_error(e),
            _ => crate::DeviceError::Unexpected, // could be unreachable!() but we prefer to be more robust
        })
    }
}

#[derive(Debug)]
struct D3D12Lib {
    lib: DynLib,
}

impl D3D12Lib {
    fn new() -> Result<Self, libloading::Error> {
        unsafe { DynLib::new("d3d12.dll").map(|lib| Self { lib }) }
    }

    fn create_device(
        &self,
        adapter: &DxgiAdapter,
        feature_level: Direct3D::D3D_FEATURE_LEVEL,
    ) -> Result<Option<Direct3D12::ID3D12Device>, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12CreateDevice on d3d12.dll
        type Fun = extern "system" fn(
            padapter: *mut core::ffi::c_void,
            minimumfeaturelevel: Direct3D::D3D_FEATURE_LEVEL,
            riid: *const windows_core::GUID,
            ppdevice: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"D3D12CreateDevice\0") }?;

        let mut result__: Option<Direct3D12::ID3D12Device> = None;

        let res = (func)(
            adapter.as_raw(),
            feature_level,
            // TODO: Generic?
            &Direct3D12::ID3D12Device::IID,
            <*mut _>::cast(&mut result__),
        )
        .ok();

        if let Err(ref err) = res {
            match err.code() {
                Dxgi::DXGI_ERROR_UNSUPPORTED => return Ok(None),
                Dxgi::DXGI_ERROR_DRIVER_INTERNAL_ERROR => return Err(crate::DeviceError::Lost),
                _ => {}
            }
        }

        res.into_device_result("Device creation")?;

        result__.ok_or(crate::DeviceError::Unexpected).map(Some)
    }

    fn serialize_root_signature(
        &self,
        version: Direct3D12::D3D_ROOT_SIGNATURE_VERSION,
        parameters: &[Direct3D12::D3D12_ROOT_PARAMETER],
        static_samplers: &[Direct3D12::D3D12_STATIC_SAMPLER_DESC],
        flags: Direct3D12::D3D12_ROOT_SIGNATURE_FLAGS,
    ) -> Result<D3DBlob, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12SerializeRootSignature on d3d12.dll
        type Fun = extern "system" fn(
            prootsignature: *const Direct3D12::D3D12_ROOT_SIGNATURE_DESC,
            version: Direct3D12::D3D_ROOT_SIGNATURE_VERSION,
            ppblob: *mut *mut core::ffi::c_void,
            pperrorblob: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> =
            unsafe { self.lib.get(b"D3D12SerializeRootSignature\0") }?;

        let desc = Direct3D12::D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as _,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: static_samplers.len() as _,
            pStaticSamplers: static_samplers.as_ptr(),
            Flags: flags,
        };

        let mut blob = None;
        let mut error = None::<Direct3D::ID3DBlob>;
        (func)(
            &desc,
            version,
            <*mut _>::cast(&mut blob),
            <*mut _>::cast(&mut error),
        )
        .ok()
        .into_device_result("Root signature serialization")?;

        if let Some(error) = error {
            let error = D3DBlob(error);
            log::error!(
                "Root signature serialization error: {:?}",
                unsafe { error.as_c_str() }.unwrap().to_str().unwrap()
            );
            return Err(crate::DeviceError::Unexpected); // could be hal_usage_error or hal_internal_error
        }

        blob.ok_or(crate::DeviceError::Unexpected)
    }

    fn debug_interface(&self) -> Result<Option<Direct3D12::ID3D12Debug>, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12GetDebugInterface on d3d12.dll
        type Fun = extern "system" fn(
            riid: *const windows_core::GUID,
            ppvdebug: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"D3D12GetDebugInterface\0") }?;

        let mut result__ = None;

        let res = (func)(&Direct3D12::ID3D12Debug::IID, <*mut _>::cast(&mut result__)).ok();

        if let Err(ref err) = res {
            match err.code() {
                Dxgi::DXGI_ERROR_SDK_COMPONENT_MISSING => return Ok(None),
                _ => {}
            }
        }

        res.into_device_result("GetDebugInterface")?;

        result__.ok_or(crate::DeviceError::Unexpected).map(Some)
    }
}

#[derive(Debug)]
pub(super) struct DxgiLib {
    lib: DynLib,
}

impl DxgiLib {
    pub fn new() -> Result<Self, libloading::Error> {
        unsafe { DynLib::new("dxgi.dll").map(|lib| Self { lib }) }
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.3 is not available.
    pub fn debug_interface1(&self) -> Result<Option<Dxgi::IDXGIInfoQueue>, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::DXGIGetDebugInterface1 on dxgi.dll
        type Fun = extern "system" fn(
            flags: u32,
            riid: *const windows_core::GUID,
            pdebug: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"DXGIGetDebugInterface1\0") }?;

        let mut result__ = None;

        let res = (func)(0, &Dxgi::IDXGIInfoQueue::IID, <*mut _>::cast(&mut result__)).ok();

        if let Err(ref err) = res {
            match err.code() {
                Dxgi::DXGI_ERROR_SDK_COMPONENT_MISSING => return Ok(None),
                _ => {}
            }
        }

        res.into_device_result("debug_interface1")?;

        result__.ok_or(crate::DeviceError::Unexpected).map(Some)
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.4 is not available.
    pub fn create_factory4(
        &self,
        factory_flags: Dxgi::DXGI_CREATE_FACTORY_FLAGS,
    ) -> Result<Dxgi::IDXGIFactory4, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::CreateDXGIFactory2 on dxgi.dll
        type Fun = extern "system" fn(
            flags: Dxgi::DXGI_CREATE_FACTORY_FLAGS,
            riid: *const windows_core::GUID,
            ppfactory: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"CreateDXGIFactory2\0") }?;

        let mut result__ = None;

        (func)(
            factory_flags,
            &Dxgi::IDXGIFactory4::IID,
            <*mut _>::cast(&mut result__),
        )
        .ok()
        .into_device_result("create_factory4")?;

        result__.ok_or(crate::DeviceError::Unexpected)
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.3 is not available.
    pub fn create_factory_media(&self) -> Result<Dxgi::IDXGIFactoryMedia, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::CreateDXGIFactory1 on dxgi.dll
        type Fun = extern "system" fn(
            riid: *const windows_core::GUID,
            ppfactory: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"CreateDXGIFactory1\0") }?;

        let mut result__ = None;

        // https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nn-dxgi1_3-idxgifactorymedia
        (func)(&Dxgi::IDXGIFactoryMedia::IID, <*mut _>::cast(&mut result__))
            .ok()
            .into_device_result("create_factory_media")?;

        result__.ok_or(crate::DeviceError::Unexpected)
    }
}

/// Create a temporary "owned" copy inside a [`mem::ManuallyDrop`] without increasing the refcount or
/// moving away the source variable.
///
/// This is a common pattern when needing to pass interface pointers ("borrows") into Windows
/// structs.  Moving/cloning ownership is impossible/inconvenient because:
///
/// - The caller does _not_ assume ownership (and decrement the refcount at a later time);
/// - Unnecessarily increasing and decrementing the refcount;
/// - [`Drop`] destructors cannot run inside `union` structures (when the created structure is
///   implicitly dropped after a call).
///
/// See also <https://github.com/microsoft/windows-rs/pull/2361#discussion_r1150799401> and
/// <https://github.com/microsoft/windows-rs/issues/2386>.
///
/// # Safety
/// Performs a [`mem::transmute_copy()`] on a refcounted [`Interface`] type.  The returned
/// [`mem::ManuallyDrop`] should _not_ be dropped.
pub unsafe fn borrow_interface_temporarily<I: Interface>(src: &I) -> mem::ManuallyDrop<Option<I>> {
    unsafe { mem::transmute_copy(src) }
}

/// See [`borrow_interface_temporarily()`]
pub unsafe fn borrow_optional_interface_temporarily<I: Interface>(
    src: &Option<I>,
) -> mem::ManuallyDrop<Option<I>> {
    unsafe { mem::transmute_copy(src) }
}

struct D3DBlob(Direct3D::ID3DBlob);

impl Deref for D3DBlob {
    type Target = Direct3D::ID3DBlob;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl D3DBlob {
    unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.GetBufferPointer().cast(), self.GetBufferSize()) }
    }

    unsafe fn as_c_str(&self) -> Result<&ffi::CStr, ffi::FromBytesUntilNulError> {
        ffi::CStr::from_bytes_until_nul(unsafe { self.as_slice() })
    }
}

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
    type PipelineCache = PipelineCache;

    type AccelerationStructure = AccelerationStructure;
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
    Texture,
    TextureView
);

// Limited by D3D12's root signature size of 64. Each element takes 1 or 2 entries.
const MAX_ROOT_ELEMENTS: usize = 64;
const ZERO_BUFFER_SIZE: wgt::BufferAddress = 256 << 10;

pub struct Instance {
    factory: DxgiFactory,
    factory_media: Option<Dxgi::IDXGIFactoryMedia>,
    library: Arc<D3D12Lib>,
    supports_allow_tearing: bool,
    _lib_dxgi: DxgiLib,
    flags: wgt::InstanceFlags,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
}

impl Instance {
    pub unsafe fn create_surface_from_visual(&self, visual: *mut ffi::c_void) -> Surface {
        let visual = unsafe { DirectComposition::IDCompositionVisual::from_raw_borrowed(&visual) }
            .expect("COM pointer should not be NULL");
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::Visual(visual.to_owned()),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: RwLock::new(None),
        }
    }

    pub unsafe fn create_surface_from_surface_handle(
        &self,
        surface_handle: *mut ffi::c_void,
    ) -> Surface {
        // TODO: We're not given ownership, so we shouldn't call HANDLE::free(). This puts an extra burden on the caller to keep it alive.
        // https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle could help us, even though DirectComposition is not in the list?
        // Or we make all these types owned, require an ownership transition, and replace SurfaceTargetUnsafe with SurfaceTarget.
        let surface_handle = Foundation::HANDLE(surface_handle);
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::SurfaceHandle(surface_handle),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: RwLock::new(None),
        }
    }

    pub unsafe fn create_surface_from_swap_chain_panel(
        &self,
        swap_chain_panel: *mut ffi::c_void,
    ) -> Surface {
        let swap_chain_panel =
            unsafe { types::ISwapChainPanelNative::from_raw_borrowed(&swap_chain_panel) }
                .expect("COM pointer should not be NULL");
        Surface {
            factory: self.factory.clone(),
            factory_media: self.factory_media.clone(),
            target: SurfaceTarget::SwapChainPanel(swap_chain_panel.to_owned()),
            supports_allow_tearing: self.supports_allow_tearing,
            swap_chain: RwLock::new(None),
        }
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

struct SwapChain {
    // TODO: Drop order frees the SWC before the raw image pointers...?
    raw: Dxgi::IDXGISwapChain3,
    // need to associate raw image pointers with the swapchain so they can be properly released
    // when the swapchain is destroyed
    resources: Vec<Direct3D12::ID3D12Resource>,
    /// Handle is freed in [`Self::release_resources()`]
    waitable: Foundation::HANDLE,
    acquired_count: usize,
    present_mode: wgt::PresentMode,
    format: wgt::TextureFormat,
    size: wgt::Extent3d,
}

enum SurfaceTarget {
    /// Borrowed, lifetime externally managed
    WndHandle(Foundation::HWND),
    Visual(DirectComposition::IDCompositionVisual),
    /// Borrowed, lifetime externally managed
    SurfaceHandle(Foundation::HANDLE),
    SwapChainPanel(types::ISwapChainPanelNative),
}

pub struct Surface {
    factory: DxgiFactory,
    factory_media: Option<Dxgi::IDXGIFactoryMedia>,
    target: SurfaceTarget,
    supports_allow_tearing: bool,
    swap_chain: RwLock<Option<SwapChain>>,
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
    heap_create_not_zeroed: bool,
    casting_fully_typed_format_supported: bool,
    suballocation_supported: bool,
    shader_model: naga::back::hlsl::ShaderModel,
}

#[derive(Default)]
struct Workarounds {
    // On WARP, temporary CPU descriptors are still used by the runtime
    // after we call `CopyDescriptors`.
    avoid_cpu_descriptor_overwrites: bool,
}

pub struct Adapter {
    raw: DxgiAdapter,
    device: Direct3D12::ID3D12Device,
    library: Arc<D3D12Lib>,
    private_caps: PrivateCapabilities,
    presentation_timer: auxil::dxgi::time::PresentationTimer,
    // Note: this isn't used right now, but we'll need it later.
    #[allow(unused)]
    workarounds: Workarounds,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

struct Event(pub Foundation::HANDLE);
impl Event {
    pub fn create(manual_reset: bool, initial_state: bool) -> Result<Self, crate::DeviceError> {
        Ok(Self(
            unsafe { Threading::CreateEventA(None, manual_reset, initial_state, None) }
                .into_device_result("CreateEventA")?,
        ))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { Foundation::HANDLE::free(&mut self.0) }
    }
}

/// Helper structure for waiting for GPU.
struct Idler {
    fence: Direct3D12::ID3D12Fence,
    event: Event,
}

struct CommandSignatures {
    draw: Direct3D12::ID3D12CommandSignature,
    draw_indexed: Direct3D12::ID3D12CommandSignature,
    dispatch: Direct3D12::ID3D12CommandSignature,
}

struct DeviceShared {
    zero_buffer: Direct3D12::ID3D12Resource,
    cmd_signatures: CommandSignatures,
    heap_views: descriptor::GeneralHeap,
    heap_samplers: descriptor::GeneralHeap,
}

unsafe impl Send for DeviceShared {}
unsafe impl Sync for DeviceShared {}

pub struct Device {
    raw: Direct3D12::ID3D12Device,
    present_queue: Direct3D12::ID3D12CommandQueue,
    idler: Idler,
    private_caps: PrivateCapabilities,
    shared: Arc<DeviceShared>,
    // CPU only pools
    rtv_pool: Mutex<descriptor::CpuPool>,
    dsv_pool: Mutex<descriptor::CpuPool>,
    srv_uav_pool: Mutex<descriptor::CpuPool>,
    sampler_pool: Mutex<descriptor::CpuPool>,
    // library
    library: Arc<D3D12Lib>,
    #[cfg(feature = "renderdoc")]
    render_doc: auxil::renderdoc::RenderDoc,
    null_rtv_handle: descriptor::Handle,
    mem_allocator: Mutex<suballocation::GpuAllocatorWrapper>,
    dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
    counters: wgt::HalCounters,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: Direct3D12::ID3D12CommandQueue,
    temp_lists: Mutex<Vec<Option<Direct3D12::ID3D12CommandList>>>,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

#[derive(Default)]
struct Temp {
    marker: Vec<u16>,
    barriers: Vec<Direct3D12::D3D12_RESOURCE_BARRIER>,
}

impl Temp {
    fn clear(&mut self) {
        self.marker.clear();
        self.barriers.clear();
    }
}

struct PassResolve {
    src: (Direct3D12::ID3D12Resource, u32),
    dst: (Direct3D12::ID3D12Resource, u32),
    format: Dxgi::Common::DXGI_FORMAT,
}

#[derive(Clone, Copy)]
enum RootElement {
    Empty,
    Constant,
    SpecialConstantBuffer {
        first_vertex: i32,
        first_instance: u32,
        other: u32,
    },
    /// Descriptor table.
    Table(Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE),
    /// Descriptor for a buffer that has dynamic offset.
    DynamicOffsetBuffer {
        kind: BufferViewKind,
        address: Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE,
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
    vertex_buffers: [Direct3D12::D3D12_VERTEX_BUFFER_VIEW; crate::MAX_VERTEX_BUFFERS],
    dirty_vertex_buffers: usize,
    kind: PassKind,
}

#[test]
fn test_dirty_mask() {
    assert_eq!(MAX_ROOT_ELEMENTS, u64::BITS as usize);
}

impl PassState {
    fn new() -> Self {
        PassState {
            has_label: false,
            resolves: ArrayVec::new(),
            layout: PipelineLayoutShared {
                signature: None,
                total_root_elements: 0,
                special_constants_root_index: None,
                root_constant_info: None,
            },
            root_elements: [RootElement::Empty; MAX_ROOT_ELEMENTS],
            constant_data: [0; MAX_ROOT_ELEMENTS],
            dirty_root_elements: 0,
            vertex_buffers: [Default::default(); crate::MAX_VERTEX_BUFFERS],
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
    allocator: Direct3D12::ID3D12CommandAllocator,
    device: Direct3D12::ID3D12Device,
    shared: Arc<DeviceShared>,
    null_rtv_handle: descriptor::Handle,
    list: Option<Direct3D12::ID3D12GraphicsCommandList>,
    free_lists: Vec<Direct3D12::ID3D12GraphicsCommandList>,
    pass: PassState,
    temp: Temp,

    /// If set, the end of the next render/compute pass will write a timestamp at
    /// the given pool & location.
    end_of_pass_timer_query: Option<(Direct3D12::ID3D12QueryHeap, u32)>,
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
    raw: Direct3D12::ID3D12GraphicsCommandList,
}

impl crate::DynCommandBuffer for CommandBuffer {}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

#[derive(Debug)]
pub struct Buffer {
    resource: Direct3D12::ID3D12Resource,
    size: wgt::BufferAddress,
    allocation: Option<suballocation::AllocationWrapper>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl crate::DynBuffer for Buffer {}

impl crate::BufferBinding<'_, Buffer> {
    fn resolve_size(&self) -> wgt::BufferAddress {
        match self.size {
            Some(size) => size.get(),
            None => self.buffer.size - self.offset,
        }
    }

    // TODO: Return GPU handle directly?
    fn resolve_address(&self) -> wgt::BufferAddress {
        (unsafe { self.buffer.resource.GetGPUVirtualAddress() }) + self.offset
    }
}

#[derive(Debug)]
pub struct Texture {
    resource: Direct3D12::ID3D12Resource,
    format: wgt::TextureFormat,
    dimension: wgt::TextureDimension,
    size: wgt::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    allocation: Option<suballocation::AllocationWrapper>,
}

impl crate::DynTexture for Texture {}
impl crate::DynSurfaceTexture for Texture {}

impl std::borrow::Borrow<dyn crate::DynTexture> for Texture {
    fn borrow(&self) -> &dyn crate::DynTexture {
        self
    }
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

    /// see <https://learn.microsoft.com/en-us/windows/win32/direct3d12/subresources#plane-slice>
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
    raw_format: Dxgi::Common::DXGI_FORMAT,
    aspects: crate::FormatAspects,
    /// only used by resolve
    target_base: (Direct3D12::ID3D12Resource, u32),
    handle_srv: Option<descriptor::Handle>,
    handle_uav: Option<descriptor::Handle>,
    handle_rtv: Option<descriptor::Handle>,
    handle_dsv_ro: Option<descriptor::Handle>,
    handle_dsv_rw: Option<descriptor::Handle>,
}

impl crate::DynTextureView for TextureView {}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

#[derive(Debug)]
pub struct Sampler {
    handle: descriptor::Handle,
}

impl crate::DynSampler for Sampler {}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

#[derive(Debug)]
pub struct QuerySet {
    raw: Direct3D12::ID3D12QueryHeap,
    raw_ty: Direct3D12::D3D12_QUERY_TYPE,
}

impl crate::DynQuerySet for QuerySet {}

unsafe impl Send for QuerySet {}
unsafe impl Sync for QuerySet {}

#[derive(Debug)]
pub struct Fence {
    raw: Direct3D12::ID3D12Fence,
}

impl crate::DynFence for Fence {}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub fn raw_fence(&self) -> &Direct3D12::ID3D12Fence {
        &self.raw
    }
}

#[derive(Debug)]
pub struct BindGroupLayout {
    /// Sorted list of entries.
    entries: Vec<wgt::BindGroupLayoutEntry>,
    cpu_heap_views: Option<descriptor::CpuHeap>,
    cpu_heap_samplers: Option<descriptor::CpuHeap>,
    copy_counts: Vec<u32>, // all 1's
}

impl crate::DynBindGroupLayout for BindGroupLayout {}

#[derive(Debug, Clone, Copy)]
enum BufferViewKind {
    Constant,
    ShaderResource,
    UnorderedAccess,
}

#[derive(Debug)]
pub struct BindGroup {
    handle_views: Option<descriptor::DualHandle>,
    handle_samplers: Option<descriptor::DualHandle>,
    dynamic_buffers: Vec<Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE>,
}

impl crate::DynBindGroup for BindGroup {}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct TableTypes: u8 {
        const SRV_CBV_UAV = 1 << 0;
        const SAMPLERS = 1 << 1;
    }
}

// Element (also known as parameter) index into the root signature.
type RootIndex = u32;

#[derive(Debug)]
struct BindGroupInfo {
    base_root_index: RootIndex,
    tables: TableTypes,
    dynamic_buffers: Vec<BufferViewKind>,
}

#[derive(Debug, Clone)]
struct RootConstantInfo {
    root_index: RootIndex,
    range: std::ops::Range<u32>,
}

#[derive(Debug, Clone)]
struct PipelineLayoutShared {
    signature: Option<Direct3D12::ID3D12RootSignature>,
    total_root_elements: RootIndex,
    special_constants_root_index: Option<RootIndex>,
    root_constant_info: Option<RootConstantInfo>,
}

unsafe impl Send for PipelineLayoutShared {}
unsafe impl Sync for PipelineLayoutShared {}

#[derive(Debug)]
pub struct PipelineLayout {
    shared: PipelineLayoutShared,
    // Storing for each associated bind group, which tables we created
    // in the root signature. This is required for binding descriptor sets.
    bind_group_infos: ArrayVec<BindGroupInfo, { crate::MAX_BIND_GROUPS }>,
    naga_options: naga::back::hlsl::Options,
}

impl crate::DynPipelineLayout for PipelineLayout {}

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
    raw_name: Option<ffi::CString>,
}

impl crate::DynShaderModule for ShaderModule {}

pub(super) enum CompiledShader {
    Dxc(Direct3D::Dxc::IDxcBlob),
    Fxc(Direct3D::ID3DBlob),
}

impl CompiledShader {
    fn create_native_shader(&self) -> Direct3D12::D3D12_SHADER_BYTECODE {
        match self {
            CompiledShader::Dxc(shader) => Direct3D12::D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { shader.GetBufferPointer() },
                BytecodeLength: unsafe { shader.GetBufferSize() },
            },
            CompiledShader::Fxc(shader) => Direct3D12::D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { shader.GetBufferPointer() },
                BytecodeLength: unsafe { shader.GetBufferSize() },
            },
        }
    }

    unsafe fn destroy(self) {}
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: Direct3D12::ID3D12PipelineState,
    layout: PipelineLayoutShared,
    topology: Direct3D::D3D_PRIMITIVE_TOPOLOGY,
    vertex_strides: [Option<NonZeroU32>; crate::MAX_VERTEX_BUFFERS],
}

impl crate::DynRenderPipeline for RenderPipeline {}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: Direct3D12::ID3D12PipelineState,
    layout: PipelineLayoutShared,
}

impl crate::DynComputePipeline for ComputePipeline {}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

#[derive(Debug)]
pub struct PipelineCache;

impl crate::DynPipelineCache for PipelineCache {}

#[derive(Debug)]
pub struct AccelerationStructure {}

impl crate::DynAccelerationStructure for AccelerationStructure {}

impl SwapChain {
    unsafe fn release_resources(mut self) -> Dxgi::IDXGISwapChain3 {
        unsafe { Foundation::HANDLE::free(&mut self.waitable) };
        self.raw
    }

    unsafe fn wait(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<bool, crate::SurfaceError> {
        let timeout_ms = match timeout {
            Some(duration) => duration.as_millis() as u32,
            None => Threading::INFINITE,
        };
        match unsafe { Threading::WaitForSingleObject(self.waitable, timeout_ms) } {
            Foundation::WAIT_ABANDONED | Foundation::WAIT_FAILED => Err(crate::SurfaceError::Lost),
            Foundation::WAIT_OBJECT_0 => Ok(true),
            Foundation::WAIT_TIMEOUT => Ok(false),
            other => {
                log::error!("Unexpected wait status: 0x{:x?}", other);
                Err(crate::SurfaceError::Lost)
            }
        }
    }
}

impl crate::Surface for Surface {
    type A = Api;

    unsafe fn configure(
        &self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let mut flags = Dxgi::DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        // We always set ALLOW_TEARING on the swapchain no matter
        // what kind of swapchain we want because ResizeBuffers
        // cannot change the swapchain's ALLOW_TEARING flag.
        //
        // This does not change the behavior of the swapchain, just
        // allow present calls to use tearing.
        if self.supports_allow_tearing {
            flags |= Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }

        // While `configure`s contract ensures that no work on the GPU's main queues
        // are in flight, we still need to wait for the present queue to be idle.
        unsafe { device.wait_for_present_queue_idle() }?;

        let non_srgb_format = auxil::dxgi::conv::map_texture_format_nosrgb(config.format);

        // The range for `SetMaximumFrameLatency` is 1-16 so the maximum latency requested should be 15 because we add 1.
        // https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgidevice1-setmaximumframelatency
        debug_assert!(config.maximum_frame_latency <= 15);

        // Nvidia recommends to use 1-2 more buffers than the maximum latency
        // https://developer.nvidia.com/blog/advanced-api-performance-swap-chains/
        // For high latency extra buffers seems excessive, so go with a minimum of 3 and beyond that add 1.
        let swap_chain_buffer = (config.maximum_frame_latency + 1).min(16);

        let swap_chain = match self.swap_chain.write().take() {
            //Note: this path doesn't properly re-initialize all of the things
            Some(sc) => {
                let raw = unsafe { sc.release_resources() };
                let result = unsafe {
                    raw.ResizeBuffers(
                        swap_chain_buffer,
                        config.extent.width,
                        config.extent.height,
                        non_srgb_format,
                        flags,
                    )
                };
                if let Err(err) = result {
                    log::error!("ResizeBuffers failed: {err}");
                    return Err(crate::SurfaceError::Other("window is in use"));
                }
                raw
            }
            None => {
                let desc = Dxgi::DXGI_SWAP_CHAIN_DESC1 {
                    AlphaMode: auxil::dxgi::conv::map_acomposite_alpha_mode(
                        config.composite_alpha_mode,
                    ),
                    Width: config.extent.width,
                    Height: config.extent.height,
                    Format: non_srgb_format,
                    Stereo: false.into(),
                    SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    BufferUsage: Dxgi::DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: swap_chain_buffer,
                    Scaling: Dxgi::DXGI_SCALING_STRETCH,
                    SwapEffect: Dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    Flags: flags.0 as u32,
                };
                let swap_chain1 = match self.target {
                    SurfaceTarget::Visual(_) | SurfaceTarget::SwapChainPanel(_) => {
                        profiling::scope!("IDXGIFactory2::CreateSwapChainForComposition");
                        unsafe {
                            self.factory.CreateSwapChainForComposition(
                                &device.present_queue,
                                &desc,
                                None,
                            )
                        }
                    }
                    SurfaceTarget::SurfaceHandle(handle) => {
                        profiling::scope!(
                            "IDXGIFactoryMedia::CreateSwapChainForCompositionSurfaceHandle"
                        );
                        unsafe {
                            self.factory_media
                                .as_ref()
                                .ok_or(crate::SurfaceError::Other("IDXGIFactoryMedia not found"))?
                                .CreateSwapChainForCompositionSurfaceHandle(
                                    &device.present_queue,
                                    handle,
                                    &desc,
                                    None,
                                )
                        }
                    }
                    SurfaceTarget::WndHandle(hwnd) => {
                        profiling::scope!("IDXGIFactory2::CreateSwapChainForHwnd");
                        unsafe {
                            self.factory.CreateSwapChainForHwnd(
                                &device.present_queue,
                                hwnd,
                                &desc,
                                None,
                                None,
                            )
                        }
                    }
                };

                let swap_chain1 = swap_chain1.map_err(|err| {
                    log::error!("SwapChain creation error: {}", err);
                    crate::SurfaceError::Other("swapchain creation")
                })?;

                match &self.target {
                    SurfaceTarget::WndHandle(_) | SurfaceTarget::SurfaceHandle(_) => {}
                    SurfaceTarget::Visual(visual) => {
                        if let Err(err) = unsafe { visual.SetContent(&swap_chain1) } {
                            log::error!("Unable to SetContent: {err}");
                            return Err(crate::SurfaceError::Other(
                                "IDCompositionVisual::SetContent",
                            ));
                        }
                    }
                    SurfaceTarget::SwapChainPanel(swap_chain_panel) => {
                        if let Err(err) = unsafe { swap_chain_panel.SetSwapChain(&swap_chain1) } {
                            log::error!("Unable to SetSwapChain: {err}");
                            return Err(crate::SurfaceError::Other(
                                "ISwapChainPanelNative::SetSwapChain",
                            ));
                        }
                    }
                }

                swap_chain1.cast::<Dxgi::IDXGISwapChain3>().map_err(|err| {
                    log::error!("Unable to cast swapchain: {err}");
                    crate::SurfaceError::Other("swapchain cast to version 3")
                })?
            }
        };

        match self.target {
            SurfaceTarget::WndHandle(wnd_handle) => {
                // Disable automatic Alt+Enter handling by DXGI.
                unsafe {
                    self.factory.MakeWindowAssociation(
                        wnd_handle,
                        Dxgi::DXGI_MWA_NO_WINDOW_CHANGES | Dxgi::DXGI_MWA_NO_ALT_ENTER,
                    )
                }
                .into_device_result("MakeWindowAssociation")?;
            }
            SurfaceTarget::Visual(_)
            | SurfaceTarget::SurfaceHandle(_)
            | SurfaceTarget::SwapChainPanel(_) => {}
        }

        unsafe { swap_chain.SetMaximumFrameLatency(config.maximum_frame_latency) }
            .into_device_result("SetMaximumFrameLatency")?;
        let waitable = unsafe { swap_chain.GetFrameLatencyWaitableObject() };

        let mut resources = Vec::with_capacity(swap_chain_buffer as usize);
        for i in 0..swap_chain_buffer {
            let resource = unsafe { swap_chain.GetBuffer(i) }
                .into_device_result("Failed to get swapchain buffer")?;
            resources.push(resource);
        }

        let mut swapchain = self.swap_chain.write();
        *swapchain = Some(SwapChain {
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

    unsafe fn unconfigure(&self, device: &Device) {
        if let Some(sc) = self.swap_chain.write().take() {
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
        &self,
        timeout: Option<std::time::Duration>,
        _fence: &Fence,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        let mut swapchain = self.swap_chain.write();
        let sc = swapchain.as_mut().unwrap();

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
    unsafe fn discard_texture(&self, _texture: Texture) {
        let mut swapchain = self.swap_chain.write();
        let sc = swapchain.as_mut().unwrap();
        sc.acquired_count -= 1;
    }
}

impl crate::Queue for Queue {
    type A = Api;

    unsafe fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        _surface_textures: &[&Texture],
        (signal_fence, signal_value): (&mut Fence, crate::FenceValue),
    ) -> Result<(), crate::DeviceError> {
        let mut temp_lists = self.temp_lists.lock();
        temp_lists.clear();
        for cmd_buf in command_buffers {
            temp_lists.push(Some(cmd_buf.raw.clone().into()));
        }

        {
            profiling::scope!("ID3D12CommandQueue::ExecuteCommandLists");
            unsafe { self.raw.ExecuteCommandLists(&temp_lists) }
        }

        unsafe { self.raw.Signal(&signal_fence.raw, signal_value) }
            .into_device_result("Signal fence")?;

        // Note the lack of synchronization here between the main Direct queue
        // and the dedicated presentation queue. This is automatically handled
        // by the D3D runtime by detecting uses of resources derived from the
        // swapchain. This automatic detection is why you cannot use a swapchain
        // as an UAV in D3D12.

        Ok(())
    }
    unsafe fn present(
        &self,
        surface: &Surface,
        _texture: Texture,
    ) -> Result<(), crate::SurfaceError> {
        let mut swapchain = surface.swap_chain.write();
        let sc = swapchain.as_mut().unwrap();
        sc.acquired_count -= 1;

        let (interval, flags) = match sc.present_mode {
            // We only allow immediate if ALLOW_TEARING is valid.
            wgt::PresentMode::Immediate => (0, Dxgi::DXGI_PRESENT_ALLOW_TEARING),
            wgt::PresentMode::Mailbox => (0, Dxgi::DXGI_PRESENT::default()),
            wgt::PresentMode::Fifo => (1, Dxgi::DXGI_PRESENT::default()),
            m => unreachable!("Cannot make surface with present mode {m:?}"),
        };

        profiling::scope!("IDXGISwapchain3::Present");
        unsafe { sc.raw.Present(interval, flags) }
            .ok()
            .into_device_result("Present")?;

        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        let frequency = unsafe { self.raw.GetTimestampFrequency() }.expect("GetTimestampFrequency");
        (1_000_000_000.0 / frequency as f64) as f32
    }
}
