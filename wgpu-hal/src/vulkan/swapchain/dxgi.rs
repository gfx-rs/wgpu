//! DXGI flip-model interop swapchain for Vulkan on Windows.
//!
//! Using an interop D3D12 device LUID-matched to the Vulkan physical device, a DXGI swapchain
//! is made on the surface, and those buffers are then shared with Vulkan. Vulkan renders
//! directly onto the buffers, there is no intermediate texture and no copy.
//!
//! # Synchronization
//!
//! It is helpful to think of the compositor as a separate queue. D3D12 does not expose manual
//! synchronization between the presentation queue and D3D12 work. It automatically adds this
//! synchronization when a command list writes or reads a back buffer. To properly synchronize
//! Vulkan with the presentation queue, we need to create a "ladder" of synchronization between
//! the presentation and Vulkan queues.
//!
//! On acquire:
//! - We first synchronize the D3D12 queue with the presentation queue, by submitting a command
//!   list that does a no-op draw onto the current buffer.
//! - We then use a shared D3D12 fence to signal the completion of the acquire command list, and
//!   wait on that fence in Vulkan.
//!
//! The user can use the buffer in Vulkan. On present:
//! - We use a second fence to have the D3D12 queue wait for the Vulkan work.
//! - Then we submit a second D3D12 command list with a no-op draw, which will synchronize the
//!   D3D12 queue with the presentation queue.
//!
//! By using these two stages (acquire and present) we can synchronize Vulkan with the
//! presentation queue. There is overhead to this approach, in testing it is about 100us per frame,
//! and this is almost certainly faster than the next best approach of using a texture->texture copy.
//!
//! ```text
//!                             stage 1: acquire                                 stage 2: present
//!
//! presentation  ─────release─╮                                                                  ╭─flip─▶
//!                            │                                                                  │
//!                            │ (implicit sync)                                  (implicit sync) │
//!                            ▼                                                                  │
//! interop queue ─────────────[acquire list]──Signal────────────────────Wait──[present list]──Present
//!                                               │                       ▲
//!                              interop_progress │                       │ vulkan_progress
//!                                               ▼                       │
//! Vulkan queue  ───────────────────────────────Wait────Vulkan Work────Signal
//! ```

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{
    any::Any,
    mem::ManuallyDrop,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use ash::vk;
use wgpu_sync::Mutex;
use windows::{
    core::{Interface as _, PCWSTR},
    Win32::{
        Foundation::{GENERIC_ALL, HANDLE, LUID, RECT},
        Graphics::{
            Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST},
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
    },
};

use crate::{
    auxil::dxgi::{
        self, d3d12_lib::D3D12Lib, dcomp::DCompLib, device_factory::DeviceFactory,
        dxgi_lib::DxgiLib, factory::DxgiFactory, handles::OwnedHandle, result::HResult as _,
        swapchain::SurfaceTarget,
    },
    vulkan::{
        semaphore_list::SemaphoreType,
        swapchain::{
            Surface, SurfaceTextureMetadata, Swapchain, SwapchainSubmissionSemaphoreGuard,
        },
        DeviceShared, InstanceShared,
    },
    Device,
};

/// Precompiled DXIL for the sync lists' no-op draw. See [`dxgi_sync.hlsl`] for the source and the
/// `dxc` invocation that regenerates these blobs. The vertex-shader container also carries the
/// (empty) root signature the pipeline is built against.
///
/// [`dxgi_sync.hlsl`]: ./dxgi_sync.hlsl
static NOOP_DRAW_VS: &[u8] = include_bytes!("dxgi_sync_vs.cso");
static NOOP_DRAW_PS: &[u8] = include_bytes!("dxgi_sync_ps.cso");

/// Pipeline for the sync lists' no-op draw: a degenerate triangle that covers no pixels, letting
/// a list reference the back buffer without altering its contents (see module docs).
struct NoopDrawPipeline {
    root_signature: ID3D12RootSignature,
    pipeline_state: ID3D12PipelineState,
}

impl NoopDrawPipeline {
    /// Builds the pipeline for back buffers whose render-target view has `format`.
    fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> Result<Self, crate::DeviceError> {
        // The vertex-shader container embeds the root signature, so build it straight from the blob.
        let root_signature: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, NOOP_DRAW_VS) }
                .into_device_result("ID3D12Device::CreateRootSignature")?;

        // A blend target with valid (blend-disabled) enum values; the degenerate draw writes nothing
        // regardless, but the runtime still validates these fields.
        let blend_target = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let no_stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };

        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rtv_formats[0] = format;

        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // Non-owning borrow of `root_signature`; `desc` is consumed by the call below, well
            // before `root_signature` drops.
            pRootSignature: unsafe { core::mem::transmute_copy(&root_signature) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: NOOP_DRAW_VS.as_ptr().cast(),
                BytecodeLength: NOOP_DRAW_VS.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: NOOP_DRAW_PS.as_ptr().cast(),
                BytecodeLength: NOOP_DRAW_PS.len(),
            },
            DS: D3D12_SHADER_BYTECODE::default(),
            HS: D3D12_SHADER_BYTECODE::default(),
            GS: D3D12_SHADER_BYTECODE::default(),
            StreamOutput: D3D12_STREAM_OUTPUT_DESC::default(),
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [blend_target; 8],
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                FrontCounterClockwise: false.into(),
                DepthBias: 0,
                DepthBiasClamp: 0.0,
                SlopeScaledDepthBias: 0.0,
                DepthClipEnable: true.into(),
                MultisampleEnable: false.into(),
                AntialiasedLineEnable: false.into(),
                ForcedSampleCount: 0,
                ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                StencilEnable: false.into(),
                StencilReadMask: 0,
                StencilWriteMask: 0,
                FrontFace: no_stencil,
                BackFace: no_stencil,
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC::default(),
            IBStripCutValue: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: rtv_formats,
            DSVFormat: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        let pipeline_state: ID3D12PipelineState =
            unsafe { device.CreateGraphicsPipelineState(&desc) }
                .into_device_result("ID3D12Device::CreateGraphicsPipelineState")?;

        Ok(Self {
            root_signature,
            pipeline_state,
        })
    }
}

/// Builds a single-resource transition barrier. `resource` is borrowed without changing its
/// refcount; the returned barrier is transient and must not outlive `resource`.
fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // `ManuallyDrop` (never dropped) holds a borrowed, non-owning copy of the interface, so
            // building the barrier does not touch the resource's refcount.
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { core::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Encodes and submits the D3D12 command list.
///
/// # Safety
///
/// `allocator` and `list` must not have any prior recording still executing on the GPU.
#[allow(clippy::too_many_arguments)]
unsafe fn execute_sync_command_list(
    queue: &ID3D12CommandQueue,
    allocator: &ID3D12CommandAllocator,
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pipeline: &NoopDrawPipeline,
    extent: wgt::Extent3d,
    final_state: D3D12_RESOURCE_STATES,
) -> Result<(), crate::DeviceError> {
    unsafe {
        allocator
            .Reset()
            .into_device_result("ID3D12CommandAllocator::Reset")?;
        list.Reset(allocator, Some(&pipeline.pipeline_state))
            .into_device_result("ID3D12GraphicsCommandList::Reset")?;

        let to_rt = transition_barrier(
            resource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        list.ResourceBarrier(&[to_rt]);

        // The no-op draw.
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: extent.width as f32,
            Height: extent.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: extent.width as i32,
            bottom: extent.height as i32,
        };
        list.OMSetRenderTargets(1, Some(core::ptr::from_ref(&rtv)), false, None);
        list.RSSetViewports(core::slice::from_ref(&viewport));
        list.RSSetScissorRects(core::slice::from_ref(&scissor));
        list.SetGraphicsRootSignature(&pipeline.root_signature);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.DrawInstanced(3, 1, 0, 0);

        let to_final =
            transition_barrier(resource, D3D12_RESOURCE_STATE_RENDER_TARGET, final_state);
        list.ResourceBarrier(&[to_final]);
        list.Close()
            .into_device_result("ID3D12GraphicsCommandList::Close")?;

        let list: ID3D12CommandList = list.cast().into_device_result("cast ID3D12CommandList")?;
        queue.ExecuteCommandLists(&[Some(list)]);
    }
    Ok(())
}

/// Instance-wide DXGI/D3D12 state for the interop swapchain, built once at instance creation and
/// stored on [`InstanceShared`].
pub(crate) struct DxgiInstance {
    /// Keeps `dxgi.dll` loaded for the lifetime of the factory.
    _lib: DxgiLib,
    /// Runtime-loaded `d3d12.dll` entry points used to create the interop D3D12 device.
    d3d12_lib: Arc<D3D12Lib>,
    /// Creates the interop D3D12 device, using an Agility SDK independent device when configured
    /// (shared with the DX12 backend).
    device_factory: DeviceFactory,
    factory: DxgiFactory,
    dcomp_lib: Arc<DCompLib>,
    supports_allow_tearing: bool,
}

// COM objects; the instance serializes access where required.
unsafe impl Send for DxgiInstance {}
unsafe impl Sync for DxgiInstance {}

/// Resolves the requested swapchain kind at instance creation, building the [`DxgiInstance`] when
/// a DXGI kind is requested. A requested DXGI kind falls back to `Native` (with a warning) when
/// DXGI or D3D12 is unavailable.
pub(crate) fn init_dxgi_instance(
    kind: wgt::VulkanSwapchainKind,
    agility_sdk: Option<&wgt::Dx12AgilitySDK>,
    flags: wgt::InstanceFlags,
) -> (wgt::VulkanSwapchainKind, Option<DxgiInstance>) {
    if kind == wgt::VulkanSwapchainKind::Native {
        log::debug!("Vulkan swapchain kind: {kind:?}");
        return (kind, None);
    }

    match dxgi::factory::create_factory(flags) {
        Ok((lib, factory)) => {
            let d3d12_lib = match D3D12Lib::new() {
                Ok(d3d12_lib) => Arc::new(d3d12_lib),
                Err(err) => {
                    log::warn!(
                        "Vulkan swapchain kind: {kind:?} requested but d3d12.dll is unavailable \
                         ({err}); falling back to Native"
                    );
                    return (wgt::VulkanSwapchainKind::Native, None);
                }
            };
            let device_factory = match DeviceFactory::new(&d3d12_lib, agility_sdk) {
                Ok(device_factory) => device_factory,
                Err(err) => {
                    log::warn!(
                        "Vulkan swapchain kind: {kind:?} requested but the D3D12 device factory \
                         could not be created ({err}); falling back to Native"
                    );
                    return (wgt::VulkanSwapchainKind::Native, None);
                }
            };
            device_factory.enable_debug_layer(&d3d12_lib, flags);
            let supports_allow_tearing = dxgi::swapchain::supports_allow_tearing(&factory);
            log::debug!(
                "Vulkan swapchain kind: {kind:?} (DXGI interop, allow_tearing: {supports_allow_tearing})"
            );
            (
                kind,
                Some(DxgiInstance {
                    _lib: lib,
                    d3d12_lib,
                    device_factory,
                    factory,
                    dcomp_lib: Arc::new(DCompLib::new()),
                    supports_allow_tearing,
                }),
            )
        }
        Err(err) => {
            log::warn!(
                "Vulkan swapchain kind: {kind:?} requested but DXGI is unavailable ({err}); \
                 falling back to Native"
            );
            (wgt::VulkanSwapchainKind::Native, None)
        }
    }
}

/// The D3D12 device that owns and presents the swapchain, cached on [`DeviceShared`] and shared
/// by all DXGI surfaces configured against that device. The swapchain is created on `queue`,
/// which also runs the acquire- and present-time sync lists.
pub(crate) struct InteropDevice {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
}

// COM objects; the D3D12 device and command queue are free-threaded.
unsafe impl Send for InteropDevice {}
unsafe impl Sync for InteropDevice {}

impl InteropDevice {
    /// Creates a shared D3D12 fence and an NT handle to it for import into Vulkan as a timeline
    /// semaphore.
    fn create_shared_fence(&self) -> Result<(ID3D12Fence, OwnedHandle), crate::DeviceError> {
        let fence: ID3D12Fence = unsafe { self.device.CreateFence(0, D3D12_FENCE_FLAG_SHARED) }
            .into_device_result("ID3D12Device::CreateFence")?;

        let handle = unsafe {
            self.device
                .CreateSharedHandle(&fence, None, GENERIC_ALL.0, PCWSTR::null())
        }
        .into_device_result("ID3D12Device::CreateSharedHandle (fence)")?;

        Ok((fence, OwnedHandle(handle)))
    }
}

fn create_interop_device(device: &DeviceShared) -> Result<InteropDevice, crate::DeviceError> {
    let luid = device.private_caps.device_luid.ok_or_else(|| {
        log::error!("Vulkan physical device does not report a LUID; cannot match a DXGI adapter");
        crate::DeviceError::Unexpected
    })?;

    let dxgi_instance = device
        .instance
        .dxgi_instance
        .as_ref()
        .ok_or(crate::DeviceError::Unexpected)?;

    let adapter = dxgi::factory::enumerate_adapters(dxgi_instance.factory.clone())
        .into_iter()
        .find(|adapter| {
            let desc = match unsafe { adapter.GetDesc1() } {
                Ok(desc) => desc,
                Err(_) => return false,
            };
            luid == luid_bytes(desc.AdapterLuid)
        })
        .ok_or_else(|| {
            log::error!("No DXGI adapter matches the Vulkan device LUID");
            crate::DeviceError::Unexpected
        })?;

    // The debug layer (when requested) was already enabled on the factory at instance creation.
    let d3d_device = dxgi_instance
        .device_factory
        .create_device(&dxgi_instance.d3d12_lib, &adapter, D3D_FEATURE_LEVEL_11_0)
        .map_err(|err| {
            log::error!("Failed to create the interop D3D12 device: {err}");
            crate::DeviceError::Unexpected
        })?;

    // The sync lists draw with DXIL (Shader Model 6) shaders, so the interop
    // device must support Shader Model 6.0. Reject it here rather than at opaque
    // pipeline-creation time.
    if !supports_shader_model_6(&d3d_device) {
        log::error!(
            "The interop D3D12 device does not support Shader Model 6.0 (DXIL), which the DXGI \
             interop swapchain requires"
        );
        return Err(crate::DeviceError::Unexpected);
    }

    let queue: ID3D12CommandQueue = unsafe {
        d3d_device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: 0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .into_device_result("ID3D12Device::CreateCommandQueue")?;

    Ok(InteropDevice {
        device: d3d_device,
        queue,
    })
}

fn luid_bytes(luid: LUID) -> [u8; vk::LUID_SIZE] {
    let mut bytes = [0u8; vk::LUID_SIZE];
    bytes[0..4].copy_from_slice(&luid.LowPart.to_ne_bytes());
    bytes[4..8].copy_from_slice(&luid.HighPart.to_ne_bytes());
    bytes
}

/// Whether `device` supports Shader Model 6.0 (DXIL). Querying `6_0` clamps the reported model to
/// the request, so success with `>= 6_0` means the device supports at least 6.0; a runtime too old
/// to recognize 6.0 fails the query and is likewise treated as unsupported.
fn supports_shader_model_6(device: &ID3D12Device) -> bool {
    let mut shader_model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_0,
    };
    let queried = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            <*mut _>::cast(&mut shader_model),
            size_of_val(&shader_model) as u32,
        )
    };
    queried.is_ok() && shader_model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_0.0
}

/// A D3D12 fence shared with Vulkan, where the same monotonic counter is visible as both an
/// `ID3D12Fence` (signalled / waited on the D3D12 queue) and a Vulkan timeline semaphore.
struct SharedFence {
    vk: vk::Semaphore,
    d3d: ID3D12Fence,
    _handle: OwnedHandle,
}

impl SharedFence {
    fn new(
        device: &DeviceShared,
        interop: &InteropDevice,
        name: &str,
    ) -> Result<Self, crate::DeviceError> {
        let (d3d, handle) = interop.create_shared_fence()?;
        let vk = device.new_timeline_semaphore(0, name)?;
        unsafe { device.import_timeline_semaphore_d3d12_fence(vk, handle.0)? };
        Ok(Self {
            vk,
            d3d,
            _handle: handle,
        })
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.vk, None) };
    }
}

/// Per-back-buffer cross-API synchronization state, shared between the swapchain and the surface
/// texture metadata of the texture currently in flight for that buffer.
#[derive(Debug)]
struct ImageSync {
    /// `vulkan_progress` value the current frame's Vulkan work signals; read at present so the
    /// D3D12 queue can wait on it before presenting.
    vulkan_progress_value: u64,
    /// `interop_progress` value the interop queue signals after this acquire's gated sync list;
    /// the first Vulkan submission into the buffer waits on it before writing (stage 1 — see
    /// module docs). Cleared to 0 once consumed; 0 means there is nothing to wait for.
    interop_progress_value: u64,
}

/// One DXGI flip-model back buffer, shared into Vulkan, plus the per-buffer D3D12 state needed to
/// run the acquire- and present-time sync lists.
struct BackBuffer {
    /// The back buffer imported into Vulkan; rendered into directly.
    texture: crate::vulkan::Texture,
    /// The same back buffer as a D3D12 resource (owned via `GetBuffer`).
    resource: ID3D12Resource,
    _shared_handle: OwnedHandle,
    /// Render-target view of `resource`, bound by the sync lists' no-op draw. Points into the
    /// swapchain's `rtv_heap`, which must outlive it.
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    /// Allocator + list for this buffer's acquire-time sync list. Kept separate from the present
    /// pair so each can be reset independently.
    acquire_allocator: ID3D12CommandAllocator,
    acquire_list: ID3D12GraphicsCommandList,
    /// Allocator + list for this buffer's present-time sync list.
    present_allocator: ID3D12CommandAllocator,
    present_list: ID3D12GraphicsCommandList,
    /// `interop_progress` value the last acquire of this buffer signalled; gates resetting
    /// `acquire_allocator`.
    acquire_submit: u64,
    /// `interop_progress` value the last present of this buffer signalled; gates resetting
    /// `present_allocator`.
    present_submit: u64,
    sync: Arc<Mutex<ImageSync>>,
}

pub(crate) struct DxgiSurface {
    instance: Arc<InstanceShared>,
    target: SurfaceTarget,
}

// `SurfaceTarget` holds an HWND / COM dcomp state; the surface is only used under wgpu's surface
// locking.
unsafe impl Send for DxgiSurface {}
unsafe impl Sync for DxgiSurface {}

impl DxgiSurface {
    pub(crate) fn new(instance: Arc<InstanceShared>, target: SurfaceTarget) -> Self {
        Self { instance, target }
    }

    fn dxgi_instance(&self) -> &DxgiInstance {
        self.instance
            .dxgi_instance
            .as_ref()
            .expect("DxgiSurface created without a DxgiInstance")
    }
}

impl Surface for DxgiSurface {
    fn surface_capabilities(
        &self,
        adapter: &crate::vulkan::Adapter,
    ) -> Option<crate::SurfaceCapabilities> {
        if !adapter.private_caps.can_present {
            return None;
        }

        Some(dxgi::swapchain::surface_capabilities(
            &self.target,
            self.dxgi_instance().supports_allow_tearing,
        ))
    }

    unsafe fn create_swapchain(
        &self,
        device: &crate::vulkan::Device,
        config: &crate::SurfaceConfiguration,
        provided_old_swapchain: Option<Box<dyn Swapchain>>,
    ) -> Result<Box<dyn Swapchain>, crate::SurfaceError> {
        profiling::scope!("DxgiSurface::create_swapchain");

        // `old`'s resources were already released by `Surface::configure`. Reclaim its DXGI
        // swapchain (a cloned COM ref keeps it alive past the drop below) so we can `ResizeBuffers`
        // instead of creating a second swapchain on the same HWND.
        let reused = provided_old_swapchain
            .as_ref()
            .and_then(|old| old.as_any().downcast_ref::<DxgiSwapchain>())
            .map(|old| old.swapchain.clone());
        drop(provided_old_swapchain);

        let dxgi_instance = self.dxgi_instance();
        let interop = device
            .shared
            .dxgi_interop
            .get_or_try_init(|| create_interop_device(&device.shared))?;

        let non_srgb_format = dxgi::conv::map_texture_format_nosrgb(config.format);

        let flags = dxgi::swapchain::swap_chain_flags(dxgi_instance.supports_allow_tearing, true);
        let buffer_count = (config.maximum_frame_latency + 1).clamp(2, 16);

        let swapchain = match reused {
            Some(swapchain) => {
                unsafe {
                    swapchain.ResizeBuffers(
                        buffer_count,
                        config.extent.width,
                        config.extent.height,
                        non_srgb_format,
                        flags,
                    )
                }
                .into_device_result("ResizeBuffers")?;
                swapchain
            }
            None => {
                self.create_dxgi_swapchain(config, interop, non_srgb_format, buffer_count, flags)?
            }
        };

        // Apply the color space unconditionally so reconfiguring away from HDR resets state.
        unsafe {
            swapchain.SetColorSpace1(dxgi::swapchain::map_surface_color_space(config.color_space))
        }
        .into_device_result("SetColorSpace1")?;

        unsafe { swapchain.SetMaximumFrameLatency(config.maximum_frame_latency) }
            .into_device_result("SetMaximumFrameLatency")?;
        let waitable = OwnedHandle(unsafe { swapchain.GetFrameLatencyWaitableObject() });

        // Vulkan will view the back buffers as SRGB through mutable format views and D3D12 backbuffers
        // are always created as typeless.
        let base_format = config.format.remove_srgb_suffix();
        let mut view_formats = config.view_formats.clone();
        if config.format != base_format && !view_formats.contains(&config.format) {
            view_formats.push(config.format);
        }

        // Pipeline + descriptor heap for the sync lists' no-op draw. The pipeline targets the
        // back buffers' (non-sRGB) format; the RTV heap holds one render-target view per buffer.
        let noop_pipeline = NoopDrawPipeline::new(&interop.device, non_srgb_format)?;
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            interop
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: buffer_count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .into_device_result("CreateDescriptorHeap (RTV)")?;
        let rtv_increment = unsafe {
            interop
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        } as usize;
        let rtv_heap_start = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };

        let mut buffers = Vec::with_capacity(buffer_count as usize);
        for i in 0..buffer_count {
            let resource: ID3D12Resource = unsafe { swapchain.GetBuffer(i) }
                .into_device_result("IDXGISwapChain::GetBuffer")?;
            let handle = unsafe {
                interop
                    .device
                    .CreateSharedHandle(&resource, None, GENERIC_ALL.0, PCWSTR::null())
            }
            .into_device_result("ID3D12Device::CreateSharedHandle (back buffer)")?;
            let handle = OwnedHandle(handle);

            let desc = crate::TextureDescriptor {
                label: None,
                size: config.extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgt::TextureDimension::D2,
                format: base_format,
                usage: config.usage,
                memory_flags: crate::MemoryFlags::empty(),
                view_formats: view_formats.clone(),
            };
            let texture = unsafe {
                device.texture_from_shared_handle(
                    handle.0,
                    vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE,
                    &desc,
                )
            }?;

            // Generate RTVs so we can render to them.
            let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_heap_start.ptr + i as usize * rtv_increment,
            };
            unsafe { interop.device.CreateRenderTargetView(&resource, None, rtv) };

            // One (allocator, list) pair for the acquire-time sync list and one for the
            // present-time sync list. Created lists start open; close them so each acquire/present
            // can uniformly reset then record.
            let make_list = || -> Result<
                (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
                crate::DeviceError,
            > {
                let allocator: ID3D12CommandAllocator = unsafe {
                    interop
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                }
                .into_device_result("CreateCommandAllocator")?;
                let list: ID3D12GraphicsCommandList = unsafe {
                    interop.device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        &allocator,
                        None,
                    )
                }
                .into_device_result("CreateCommandList")?;
                unsafe { list.Close() }.into_device_result("ID3D12GraphicsCommandList::Close")?;
                Ok((allocator, list))
            };
            let (acquire_allocator, acquire_list) = make_list()?;
            let (present_allocator, present_list) = make_list()?;

            buffers.push(BackBuffer {
                texture,
                resource,
                _shared_handle: handle,
                rtv,
                acquire_allocator,
                acquire_list,
                present_allocator,
                present_list,
                acquire_submit: 0,
                present_submit: 0,
                sync: Arc::new(Mutex::new(ImageSync {
                    vulkan_progress_value: 0,
                    interop_progress_value: 0,
                })),
            });
        }

        let vulkan_progress =
            SharedFence::new(&device.shared, interop, "DXGI swapchain vulkan_progress")?;
        let interop_progress =
            SharedFence::new(&device.shared, interop, "DXGI swapchain interop_progress")?;

        log::debug!(
            "Configured DXGI interop swapchain: target {}, {} back buffers, format {:?}, present mode {:?}",
            match self.target {
                SurfaceTarget::WndHandle(_) => "HWND",
                SurfaceTarget::VisualFromWndHandle { .. } => "composition visual",
                #[allow(unreachable_patterns)]
                _ => "other",
            },
            buffers.len(),
            config.format,
            config.present_mode,
        );

        Ok(Box::new(DxgiSwapchain {
            device: Arc::clone(&device.shared),
            swapchain,
            waitable: Some(waitable),
            present_mode: config.present_mode,
            config: config.clone(),
            buffers,
            noop_pipeline,
            _rtv_heap: rtv_heap,
            vulkan_progress,
            vulkan_progress_counter: Arc::new(AtomicU64::new(0)),
            queue: interop.queue.clone(),
            interop_progress,
            interop_progress_value: 0,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DxgiSurface {
    fn create_dxgi_swapchain(
        &self,
        config: &crate::SurfaceConfiguration,
        interop: &InteropDevice,
        non_srgb_format: DXGI_FORMAT,
        buffer_count: u32,
        flags: DXGI_SWAP_CHAIN_FLAG,
    ) -> Result<IDXGISwapChain3, crate::SurfaceError> {
        let dxgi_instance = self.dxgi_instance();
        let desc = dxgi::swapchain::swap_chain_descriptor(
            non_srgb_format,
            config.extent,
            buffer_count,
            dxgi::conv::map_acomposite_alpha_mode(config.composite_alpha_mode),
            flags,
        );

        let swapchain1: IDXGISwapChain1 = match &self.target {
            SurfaceTarget::WndHandle(hwnd) => {
                profiling::scope!("IDXGIFactory2::CreateSwapChainForHwnd");
                unsafe {
                    dxgi_instance.factory.CreateSwapChainForHwnd(
                        &interop.queue,
                        *hwnd,
                        &desc,
                        None,
                        None,
                    )
                }
                .into_device_result("CreateSwapChainForHwnd")?
            }
            SurfaceTarget::VisualFromWndHandle { .. } => {
                profiling::scope!("IDXGIFactory2::CreateSwapChainForComposition");
                unsafe {
                    dxgi_instance
                        .factory
                        .CreateSwapChainForComposition(&interop.queue, &desc, None)
                }
                .into_device_result("CreateSwapChainForComposition")?
            }
            // The DX12-only external-surface targets are never built by the Vulkan backend.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(crate::SurfaceError::Other(
                    "unsupported DXGI surface target",
                ))
            }
        };

        match &self.target {
            SurfaceTarget::WndHandle(hwnd) => {
                // Disable DXGI's automatic Alt+Enter fullscreen handling.
                unsafe {
                    dxgi_instance.factory.MakeWindowAssociation(
                        *hwnd,
                        DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER,
                    )
                }
                .into_device_result("MakeWindowAssociation")?;
            }
            SurfaceTarget::VisualFromWndHandle {
                handle,
                dcomp_state,
            } => {
                let mut dcomp_state = dcomp_state.lock();
                let dcomp_state =
                    unsafe { dcomp_state.get_or_init(&dxgi_instance.dcomp_lib, handle) }?;
                unsafe { dcomp_state.visual.SetContent(&swapchain1) }
                    .into_device_result("IDCompositionVisual::SetContent")?;
                unsafe { dcomp_state.device.Commit() }
                    .into_device_result("IDCompositionDevice::Commit")?;
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }

        Ok(swapchain1
            .cast::<IDXGISwapChain3>()
            .into_device_result("cast IDXGISwapChain3")?)
    }
}

pub(crate) struct DxgiSwapchain {
    device: Arc<DeviceShared>,
    swapchain: IDXGISwapChain3,
    waitable: Option<OwnedHandle>,
    present_mode: wgt::PresentMode,
    config: crate::SurfaceConfiguration,

    buffers: Vec<BackBuffer>,
    /// Pipeline for the sync lists' no-op draw, shared by every back buffer.
    noop_pipeline: NoopDrawPipeline,
    /// Non-shader-visible RTV heap backing every [`BackBuffer::rtv`]; kept alive here for their sake.
    _rtv_heap: ID3D12DescriptorHeap,

    /// Shared fence Vulkan signals as each frame's rendering completes; the interop queue waits
    /// on it before presenting (stage 2 — see module docs).
    vulkan_progress: SharedFence,
    /// Allocates monotonically increasing `vulkan_progress` values; shared with each in-flight
    /// texture's metadata so the submission can stamp the value it signals.
    vulkan_progress_counter: Arc<AtomicU64>,

    /// The interop D3D12 queue (clone of [`InteropDevice::queue`]) that owns the swapchain and
    /// runs the sync lists.
    queue: ID3D12CommandQueue,
    /// Shared fence the interop queue signals (with the monotonically increasing
    /// `interop_progress_value`) after every sync list it runs. CPU-side it gates allocator reuse
    /// and teardown via [`DxgiSwapchain::wait_for_value`]; GPU-side, the value signalled after an
    /// acquire's gated list is what Vulkan waits on before rendering (stage 1 — see module docs).
    /// Vulkan never waits on the interleaved present-list values; those merely advance the fence.
    interop_progress: SharedFence,
    interop_progress_value: u64,
}

// Holds COM objects and Vulkan handles; only used under wgpu's surface locking.
unsafe impl Send for DxgiSwapchain {}
unsafe impl Sync for DxgiSwapchain {}

impl DxgiSwapchain {
    /// Blocks until the interop queue has reached `value` on `interop_progress`.
    fn wait_for_value(&self, value: u64) -> Result<(), crate::DeviceError> {
        if value == 0 {
            return Ok(());
        }
        // A null event handle makes `SetEventOnCompletion` block until the fence reaches `value`.
        unsafe {
            self.interop_progress
                .d3d
                .SetEventOnCompletion(value, HANDLE::default())
        }
        .into_device_result("SetEventOnCompletion")?;
        Ok(())
    }
}

impl Swapchain for DxgiSwapchain {
    unsafe fn release_resources(&mut self, device: &crate::vulkan::Device) {
        profiling::scope!("DxgiSwapchain::release_resources");

        // Wait for both APIs to fully drain before releasing the shared back buffers. The interop
        // queue signals `interop_progress` after `Present` and after every sync list it runs, so
        // waiting for `interop_progress_value` also waits out the last in-flight present.
        let _ = unsafe { device.shared.raw.device_wait_idle() };
        let _ = self.wait_for_value(self.interop_progress_value);

        for buffer in self.buffers.drain(..) {
            unsafe { device.destroy_texture(buffer.texture) };
            // `buffer.resource`, its command allocators/lists, and `_shared_handle` drop here.
        }
        unsafe {
            self.vulkan_progress.destroy(&device.shared.raw);
            self.interop_progress.destroy(&device.shared.raw);
        }
    }

    unsafe fn acquire(
        &mut self,
        timeout: Option<Duration>,
        _fence: &crate::vulkan::Fence,
    ) -> Result<crate::AcquiredSurfaceTexture<crate::api::Vulkan>, crate::SurfaceError> {
        if !dxgi::swapchain::wait_for_waitable(self.waitable.as_ref().map(|h| h.0), timeout)? {
            return Err(crate::SurfaceError::Timeout);
        }

        let index = unsafe { self.swapchain.GetCurrentBackBufferIndex() } as usize;

        // Make sure this buffer's previous acquire list has retired before reusing its allocator.
        self.wait_for_value(self.buffers[index].acquire_submit)?;

        // Run the acquire sync list, then signal the interop fence.
        let buffer = &self.buffers[index];
        unsafe {
            execute_sync_command_list(
                &self.queue,
                &buffer.acquire_allocator,
                &buffer.acquire_list,
                &buffer.resource,
                buffer.rtv,
                &self.noop_pipeline,
                self.config.extent,
                D3D12_RESOURCE_STATE_COMMON,
            )
        }?;
        self.interop_progress_value += 1;
        let acquire_value = self.interop_progress_value;
        unsafe { self.queue.Signal(&self.interop_progress.d3d, acquire_value) }
            .into_device_result("ID3D12CommandQueue::Signal")?;
        self.buffers[index].acquire_submit = acquire_value;

        {
            let mut sync = self.buffers[index].sync.lock();
            // Clear the `vulkan_progress` value from any prior acquisition; the submission that
            // renders this frame stamps the real value through the semaphore guard. If no work is
            // submitted, present waits on 0, which is always satisfied.
            sync.vulkan_progress_value = 0;
            // The first Vulkan submission into this buffer waits on this value before writing.
            sync.interop_progress_value = acquire_value;
        }

        let buffer = &self.buffers[index];
        let identity = self.device.texture_identity_factory.next();
        let texture = crate::vulkan::SurfaceTexture {
            index: index as u32,
            texture: crate::vulkan::Texture {
                raw: buffer.texture.raw,
                drop_guard: None,
                memory: crate::vulkan::TextureMemory::External,
                format: self.config.format,
                copy_size: buffer.texture.copy_size,
                identity,
                // D3D12 interop needs to happen through `COMMON` in D3D12 and `GENERAL` in Vulkan.
                present_layout: vk::ImageLayout::GENERAL,
            },
            metadata: Box::new(DxgiSurfaceTextureMetadata {
                vulkan_progress: self.vulkan_progress.vk,
                interop_progress: self.interop_progress.vk,
                vulkan_progress_counter: Arc::clone(&self.vulkan_progress_counter),
                sync: Arc::clone(&buffer.sync),
            }),
        };

        Ok(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal: false,
        })
    }

    unsafe fn discard_texture(
        &mut self,
        _texture: crate::vulkan::SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        // No-op: This is almost certainly not right, but is a longstanding issue with a lot of backends.
        // See https://github.com/gfx-rs/wgpu/issues/5723
        Ok(())
    }

    unsafe fn present(
        &mut self,
        _queue: &crate::vulkan::Queue,
        texture: crate::vulkan::SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let index = texture.index as usize;
        let vulkan_progress_value = self.buffers[index].sync.lock().vulkan_progress_value;
        let (sync_interval, present_flags_value) =
            dxgi::swapchain::present_flags(self.present_mode);

        // Ensure this back buffer's present allocator from its previous present is free to reset.
        self.wait_for_value(self.buffers[index].present_submit)?;

        // Order the present list (and thus the present) after this frame's Vulkan rendering.
        unsafe {
            self.queue
                .Wait(&self.vulkan_progress.d3d, vulkan_progress_value)
                .into_device_result("ID3D12CommandQueue::Wait")?;
        }
        // Run the present sync list, ending the buffer in `PRESENT` so we can present it.
        let buffer = &self.buffers[index];
        unsafe {
            execute_sync_command_list(
                &self.queue,
                &buffer.present_allocator,
                &buffer.present_list,
                &buffer.resource,
                buffer.rtv,
                &self.noop_pipeline,
                self.config.extent,
                D3D12_RESOURCE_STATE_PRESENT,
            )
        }?;

        unsafe { self.swapchain.Present(sync_interval, present_flags_value) }
            .ok()
            .into_device_result("IDXGISwapChain::Present")?;

        // This `Signal` tracks the present list's GPU completion, not the flip (no fence observes
        // the flip itself). It is what allocator reuse and `release_resources` wait on; releasing
        // the back buffer before it trips D3D12 #921 (OBJECT_DELETED_WHILE_STILL_IN_USE).
        self.interop_progress_value += 1;
        unsafe {
            self.queue
                .Signal(&self.interop_progress.d3d, self.interop_progress_value)
        }
        .into_device_result("ID3D12CommandQueue::Signal")?;
        self.buffers[index].present_submit = self.interop_progress_value;

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Debug)]
struct DxgiSurfaceTextureMetadata {
    /// The Vulkan semaphore that the Vulkan submission signals when it finishes rendering into this back buffer.
    vulkan_progress: vk::Semaphore,
    /// The Vulkan semaphore that the Vulkan submission waits on before rendering into this back buffer.
    interop_progress: vk::Semaphore,
    vulkan_progress_counter: Arc<AtomicU64>,
    sync: Arc<Mutex<ImageSync>>,
}

impl SurfaceTextureMetadata for DxgiSurfaceTextureMetadata {
    fn get_semaphore_guard(&self) -> Box<dyn SwapchainSubmissionSemaphoreGuard + '_> {
        Box::new(DxgiSwapchainSubmissionSemaphoreGuard {
            vulkan_progress: self.vulkan_progress,
            interop_progress: self.interop_progress,
            vulkan_progress_counter: &self.vulkan_progress_counter,
            sync: self.sync.lock(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct DxgiSwapchainSubmissionSemaphoreGuard<'a> {
    vulkan_progress: vk::Semaphore,
    interop_progress: vk::Semaphore,
    vulkan_progress_counter: &'a AtomicU64,
    sync: wgpu_sync::MutexGuard<'a, ImageSync>,
}

impl SwapchainSubmissionSemaphoreGuard for DxgiSwapchainSubmissionSemaphoreGuard<'_> {
    fn set_used_fence_value(&mut self, _value: u64) {
        // Unused: the DXGI path does not gate back-buffer reuse on the device fence.
    }

    fn get_acquire_wait_semaphore(&mut self) -> Option<SemaphoreType> {
        // Stage 1's cross-API wait (see module docs). Consume the value so only the first
        // submission of the frame waits; later submissions are ordered after it by the relay
        // semaphores.
        let value = core::mem::take(&mut self.sync.interop_progress_value);
        (value != 0).then_some(SemaphoreType::Timeline(self.interop_progress, value))
    }

    fn get_submit_signal_semaphore(
        &mut self,
        _device: &DeviceShared,
    ) -> Result<SemaphoreType, crate::DeviceError> {
        let value = self.vulkan_progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.sync.vulkan_progress_value = value;
        Ok(SemaphoreType::Timeline(self.vulkan_progress, value))
    }
}
