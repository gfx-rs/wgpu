/*!
# DirectX12 API internals.

## Pipeline Layout

!*/

#![allow(unused_variables)]

mod adapter;
mod command;
mod conv;
mod device;

use std::{borrow::Cow, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_4, dxgi1_6, windef, winerror},
    Interface as _,
};

#[derive(Clone)]
pub struct Api;
//TODO: remove these temporaries
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = Encoder;
    type CommandBuffer = Resource;

    type Buffer = Resource;
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

trait HResult {
    fn to_error(self) -> Option<Cow<'static, str>>;
}
impl HResult for i32 {
    fn to_error(self) -> Option<Cow<'static, str>> {
        if self >= 0 {
            return None;
        }
        let description = match self {
            winerror::E_UNEXPECTED => "unexpected",
            winerror::E_NOTIMPL => "not implemented",
            winerror::E_OUTOFMEMORY => "out of memory",
            winerror::E_INVALIDARG => "invalid argument",
            _ => return Some(Cow::Owned(format!("0x{:X}", self as u32))),
        };
        Some(Cow::Borrowed(description))
    }
}

trait HResultPair {
    type Object;
    fn check(self) -> Result<Self::Object, Cow<'static, str>>;
}
impl<T> HResultPair for (T, i32) {
    type Object = T;
    fn check(self) -> Result<T, Cow<'static, str>> {
        match self.1.to_error() {
            None => Ok(self.0),
            Some(err) => Err(err),
        }
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

struct SwapChain {}

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

pub struct Device {
    raw: native::Device,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Queue {
    raw: native::CommandQueue,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl crate::Instance<Api> for Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let lib_main = native::D3D12Lib::new().map_err(|_| crate::InstanceError)?;

        let lib_dxgi = native::DxgiLib::new().map_err(|_| crate::InstanceError)?;
        let mut factory_flags = native::FactoryCreationFlags::empty();

        if desc.flags.contains(crate::InstanceFlags::VALIDATION) {
            // Enable debug layer
            match lib_main.get_debug_interface() {
                Ok(pair) => match pair.check() {
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
                Ok(pair) => match pair.check() {
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
            Ok(pair) => match pair.check() {
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
        let factory6 = match self.factory.cast::<dxgi1_6::IDXGIFactory6>().check() {
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
                    if let Some(err) = hr.to_error() {
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
                    if let Some(err) = hr.to_error() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    match adapter1.cast::<dxgi1_2::IDXGIAdapter2>().check() {
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

impl crate::Surface<Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &Device) {}

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
