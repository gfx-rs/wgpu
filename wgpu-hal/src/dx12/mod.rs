/*!
# DirectX12 API internals.

## Pipeline Layout

!*/

#![allow(unused_variables)]

mod adapter;
mod command;
mod conv;
mod device;

use std::{borrow::Cow, ptr, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_4, dxgi1_6, dxgitype, windef, winerror},
    um::{d3d12, synchapi, winbase, winnt},
    Interface as _,
};

#[derive(Clone)]
pub struct Api;
//TODO: remove these temporaries
#[derive(Debug)]
pub struct Resource;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = CommandEncoder;
    type CommandBuffer = Resource;

    type Buffer = Buffer;
    type Texture = Resource;
    type SurfaceTexture = Resource;
    type TextureView = Resource;
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

trait HResult<O> {
    fn to_result(self) -> Result<O, Cow<'static, str>>;
    fn to_device_result(self, description: &str) -> Result<O, crate::DeviceError>;
}
impl HResult<()> for i32 {
    fn to_result(self) -> Result<(), Cow<'static, str>> {
        if self >= 0 {
            return Ok(());
        }
        let description = match self {
            winerror::E_UNEXPECTED => "unexpected",
            winerror::E_NOTIMPL => "not implemented",
            winerror::E_OUTOFMEMORY => "out of memory",
            winerror::E_INVALIDARG => "invalid argument",
            _ => return Err(Cow::Owned(format!("0x{:X}", self as u32))),
        };
        Err(Cow::Borrowed(description))
    }
    fn to_device_result(self, description: &str) -> Result<(), crate::DeviceError> {
        self.to_result().map_err(|err| {
            log::error!("{} failed: {}", description, err);
            if self == winerror::E_OUTOFMEMORY {
                crate::DeviceError::OutOfMemory
            } else {
                crate::DeviceError::Lost
            }
        })
    }
}

impl<T> HResult<T> for (T, i32) {
    fn to_result(self) -> Result<T, Cow<'static, str>> {
        self.1.to_result().map(|()| self.0)
    }
    fn to_device_result(self, description: &str) -> Result<T, crate::DeviceError> {
        self.1.to_device_result(description).map(|()| self.0)
    }
}

pub struct Instance {
    factory: native::Factory4,
    library: Arc<native::D3D12Lib>,
    lib_dxgi: native::DxgiLib,
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.factory.destroy();
        }
    }
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[derive(Copy, Clone)]
struct DualHandle {
    cpu: native::CpuDescriptor,
    gpu: native::GpuDescriptor,
    /// How large the block allocated to this handle is.
    size: u64,
}

type DescriptorIndex = u64;

struct DescriptorHeap {
    raw: native::DescriptorHeap,
    handle_size: u64,
    total_handles: u64,
    start: DualHandle,
}

impl DescriptorHeap {
    fn at(&self, index: DescriptorIndex, size: u64) -> DualHandle {
        assert!(index < self.total_handles);
        DualHandle {
            cpu: self.cpu_descriptor_at(index),
            gpu: self.gpu_descriptor_at(index),
            size,
        }
    }

    fn cpu_descriptor_at(&self, index: u64) -> native::CpuDescriptor {
        native::CpuDescriptor {
            ptr: self.start.cpu.ptr + (self.handle_size * index) as usize,
        }
    }

    fn gpu_descriptor_at(&self, index: u64) -> native::GpuDescriptor {
        native::GpuDescriptor {
            ptr: self.start.gpu.ptr + self.handle_size * index,
        }
    }
}

struct SwapChain {
    raw: native::WeakPtr<dxgi1_4::IDXGISwapChain3>,
    // need to associate raw image pointers with the swapchain so they can be properly released
    // when the swapchain is destroyed
    resources: Vec<native::Resource>,
    waitable: winnt::HANDLE,
    acquired_count: usize,
}

pub struct Surface {
    factory: native::WeakPtr<dxgi1_4::IDXGIFactory4>,
    wnd_handle: windef::HWND,
    swap_chain: Option<SwapChain>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug, Clone, Copy)]
enum MemoryArchitecture {
    Unified { cache_coherent: bool },
    NonUnified,
}

#[derive(Debug, Clone, Copy)]
struct PrivateCapabilities {
    heterogeneous_resource_heaps: bool,
    memory_architecture: MemoryArchitecture,
}

#[derive(Default)]
struct Workarounds {
    // On WARP, temporary CPU descriptors are still used by the runtime
    // after we call `CopyDescriptors`.
    avoid_cpu_descriptor_overwrites: bool,
}

pub struct Adapter {
    raw: native::WeakPtr<dxgi1_2::IDXGIAdapter2>,
    device: native::Device,
    library: Arc<native::D3D12Lib>,
    private_caps: PrivateCapabilities,
    workarounds: Workarounds,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

/// Helper structure for waiting for GPU.
struct Idler {
    fence: native::Fence,
    event: native::Event,
}

pub struct Device {
    raw: native::Device,
    present_queue: native::CommandQueue,
    idler: Idler,
    private_caps: PrivateCapabilities,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: native::CommandQueue,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

#[derive(Debug)]
pub struct Buffer {
    resource: native::Resource,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

pub struct CommandEncoder {}

impl crate::Instance<Api> for Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let lib_main = native::D3D12Lib::new().map_err(|_| crate::InstanceError)?;

        let lib_dxgi = native::DxgiLib::new().map_err(|_| crate::InstanceError)?;
        let mut factory_flags = native::FactoryCreationFlags::empty();

        if desc.flags.contains(crate::InstanceFlags::VALIDATION) {
            // Enable debug layer
            match lib_main.get_debug_interface() {
                Ok(pair) => match pair.to_result() {
                    Ok(debug_controller) => {
                        debug_controller.enable_layer();
                        debug_controller.Release();
                    }
                    Err(err) => {
                        log::warn!("Unable to enable D3D12 debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for D3D12 not found: {:?}", err);
                }
            }

            // The `DXGI_CREATE_FACTORY_DEBUG` flag is only allowed to be passed to
            // `CreateDXGIFactory2` if the debug interface is actually available. So
            // we check for whether it exists first.
            match lib_dxgi.get_debug_interface1() {
                Ok(pair) => match pair.to_result() {
                    Ok(debug_controller) => {
                        debug_controller.destroy();
                        factory_flags |= native::FactoryCreationFlags::DEBUG;
                    }
                    Err(err) => {
                        log::warn!("Unable to enable DXGI debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for DXGI not found: {:?}", err);
                }
            }
        }

        // Create DXGI factory
        let factory = match lib_dxgi.create_factory2(factory_flags) {
            Ok(pair) => match pair.to_result() {
                Ok(factory) => factory,
                Err(err) => {
                    log::warn!("Failed to create DXGI factory: {}", err);
                    return Err(crate::InstanceError);
                }
            },
            Err(err) => {
                log::warn!("Factory creation function for DXGI not found: {:?}", err);
                return Err(crate::InstanceError);
            }
        };

        Ok(Self {
            factory,
            library: Arc::new(lib_main),
            lib_dxgi,
        })
    }

    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        match has_handle.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Windows(handle) => Ok(Surface {
                factory: self.factory,
                wnd_handle: handle.hwnd as *mut _,
                swap_chain: None,
            }),
            _ => Err(crate::InstanceError),
        }
    }
    unsafe fn destroy_surface(&self, _surface: Surface) {
        // just drop
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        // Try to use high performance order by default (returns None on Windows < 1803)
        let factory6 = match self.factory.cast::<dxgi1_6::IDXGIFactory6>().to_result() {
            Ok(f6) => {
                // It's okay to decrement the refcount here because we
                // have another reference to the factory already owned by `self`.
                f6.destroy();
                Some(f6)
            }
            Err(err) => {
                log::info!("Failed to cast DXGI to 1.6: {}", err);
                None
            }
        };

        // Enumerate adapters
        let mut adapters = Vec::new();
        for cur_index in 0.. {
            let raw = match factory6 {
                Some(factory) => {
                    let mut adapter2 = native::WeakPtr::<dxgi1_2::IDXGIAdapter2>::null();
                    let hr = factory.EnumAdapterByGpuPreference(
                        cur_index,
                        dxgi1_6::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        &dxgi1_2::IDXGIAdapter2::uuidof(),
                        adapter2.mut_void(),
                    );

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Err(err) = hr.to_result() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    adapter2
                }
                None => {
                    let mut adapter1 = native::WeakPtr::<dxgi::IDXGIAdapter1>::null();
                    let hr = self
                        .factory
                        .EnumAdapters1(cur_index, adapter1.mut_void() as *mut *mut _);

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Err(err) = hr.to_result() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    match adapter1.cast::<dxgi1_2::IDXGIAdapter2>().to_result() {
                        Ok(adapter2) => {
                            adapter1.destroy();
                            adapter2
                        }
                        Err(err) => {
                            log::error!("Failed casting to Adapter2: {}", err);
                            break;
                        }
                    }
                }
            };

            adapters.extend(Adapter::expose(raw, &self.library));
        }
        adapters
    }
}

impl SwapChain {
    unsafe fn release_resources(self) -> native::WeakPtr<dxgi1_4::IDXGISwapChain3> {
        for resource in self.resources {
            resource.destroy();
        }
        self.raw
    }

    unsafe fn wait(&mut self, timeout_ms: u32) -> Result<bool, crate::SurfaceError> {
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

        let non_srgb_format = conv::map_texture_format_nosrgb(config.format);

        let swap_chain = match self.swap_chain.take() {
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
                if let Err(err) = result.to_result() {
                    log::error!("ResizeBuffers failed: {}", err);
                    return Err(crate::SurfaceError::Other("window is in use"));
                }
                raw
            }
            None => {
                let mut swap_chain1 = native::WeakPtr::<dxgi1_2::IDXGISwapChain1>::null();

                let raw_desc = dxgi1_2::DXGI_SWAP_CHAIN_DESC1 {
                    AlphaMode: conv::map_acomposite_alpha_mode(config.composite_alpha_mode),
                    BufferCount: config.swap_chain_size,
                    Width: config.extent.width,
                    Height: config.extent.height,
                    Format: non_srgb_format,
                    Flags: flags,
                    BufferUsage: dxgitype::DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Scaling: dxgi1_2::DXGI_SCALING_STRETCH,
                    Stereo: 0,
                    SwapEffect: dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
                };

                let hr = self.factory.CreateSwapChainForHwnd(
                    device.present_queue.as_mut_ptr() as *mut _,
                    self.wnd_handle,
                    &raw_desc,
                    ptr::null(),
                    ptr::null_mut(),
                    swap_chain1.mut_void() as *mut *mut _,
                );

                if let Err(err) = hr.to_result() {
                    log::error!("SwapChain creation error: {}", err);
                    return Err(crate::SurfaceError::Other("swap chain creation"));
                }

                match swap_chain1.cast::<dxgi1_4::IDXGISwapChain3>().to_result() {
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

        // Disable automatic Alt+Enter handling by DXGI.
        const DXGI_MWA_NO_WINDOW_CHANGES: u32 = 1;
        const DXGI_MWA_NO_ALT_ENTER: u32 = 2;
        self.factory.MakeWindowAssociation(
            self.wnd_handle,
            DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER,
        );

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
            //format: config.format,
            //size: config.extent,
            //mode: config.present_mode,
        });

        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &Device) {
        if let Some(mut sc) = self.swap_chain.take() {
            let _ = sc.wait(winbase::INFINITE);
            //TODO: this shouldn't be needed,
            // but it complains that the queue is still used otherwise
            let _ = device.wait_idle();
            let raw = sc.release_resources();
            raw.destroy();
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Api>>, crate::SurfaceError> {
        Ok(None)
    }
    unsafe fn discard_texture(&mut self, texture: Resource) {}
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&Resource],
        signal_fence: Option<(&mut Resource, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        Ok(())
    }
    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        texture: Resource,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }
}
