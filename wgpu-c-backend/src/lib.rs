mod adapter;
mod command;
mod conv;
mod device;
mod pass;
mod resource;
mod surface;

use std::future;
use std::pin::Pin;

use wgpu::custom::*;
use wgpu::InstanceDescriptor;
use wgpu_native::{native, *};

pub use adapter::CAdapter;
pub use command::{CCommandEncoder, CRenderBundleEncoder};
pub use device::{CDevice, CQueue};
pub use pass::{CComputePass, CRenderPass};
pub use resource::{
    CBindGroup, CBindGroupLayout, CBuffer, CBufferMappedRange, CCommandBuffer, CComputePipeline,
    CPipelineCache, CPipelineLayout, CQuerySet, CRenderBundle, CRenderPipeline, CSampler,
    CShaderModule, CTexture, CTextureView,
};
pub use surface::{CSurface, CSurfaceOutputDetail};

// ── CInstance ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CInstance {
    ptr: native::WGPUInstance,
}

unsafe impl Send for CInstance {}
unsafe impl Sync for CInstance {}

impl Drop for CInstance {
    fn drop(&mut self) {
        unsafe { wgpuInstanceRelease(self.ptr) };
    }
}

impl InstanceInterface for CInstance {
    fn new(desc: InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        let backends = conv::backends_to_native(desc.backends);
        let flags = conv::instance_flags_to_native(desc.flags);
        let dx12_compiler =
            conv::dx12_compiler_to_native(&desc.backend_options.dx12.shader_compiler);
        let dx12_presentation_system =
            conv::dx12_swapchain_kind_to_native(desc.backend_options.dx12.presentation_system);
        let gles3_minor_version =
            conv::gles3_minor_version_to_native(desc.backend_options.gl.gles_minor_version);
        let gl_fence_behaviour =
            conv::gl_fence_behavior_to_native(desc.backend_options.gl.fence_behavior);

        // Keep dxc_path alive for the duration of the C call.
        let dxc_path_str: String;
        let dxc_path = match &desc.backend_options.dx12.shader_compiler {
            wgpu::Dx12Compiler::DynamicDxc { dxc_path } => {
                dxc_path_str = dxc_path.clone();
                conv::str_to_string_view(&dxc_path_str)
            }
            _ => conv::null_string_view(),
        };

        let mut extras = native::WGPUInstanceExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_InstanceExtras,
            },
            backends,
            flags,
            dx12ShaderCompiler: dx12_compiler,
            gles3MinorVersion: gles3_minor_version,
            glFenceBehaviour: gl_fence_behaviour,
            dxcPath: dxc_path,
            dx12PresentationSystem: dx12_presentation_system,
            // SAFETY: zero is valid — budgets are optional (null = no limit),
            // displayHandle type_=0 means WGPUNativeDisplayHandleType_None.
            ..unsafe { std::mem::zeroed() }
        };

        let c_desc = native::WGPUInstanceDescriptor {
            nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut extras.chain),
            requiredFeatureCount: 0,
            requiredFeatures: std::ptr::null(),
            requiredLimits: std::ptr::null(),
        };

        let ptr = unsafe { wgpuCreateInstance(Some(&c_desc)) };
        CInstance { ptr }
    }

    unsafe fn create_surface(
        &self,
        target: wgpu::SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, wgpu::CreateSurfaceError> {
        #[allow(unused_imports)]
        use wgpu::rwh::{RawDisplayHandle, RawWindowHandle};

        let ptr = match target {
            #[allow(unused_variables)]
            wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            } => match raw_window_handle {
                #[cfg(target_os = "macos")]
                RawWindowHandle::AppKit(h) => {
                    let mut src = native::WGPUSurfaceSourceMetalLayer {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceMetalLayer,
                        },
                        layer: h.ns_view.as_ptr(),
                    };
                    let c_desc = native::WGPUSurfaceDescriptor {
                        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                            &mut src.chain,
                        ),
                        label: conv::null_string_view(),
                    };
                    unsafe { wgpuInstanceCreateSurface(self.ptr, Some(&c_desc)) }
                }
                #[cfg(target_os = "windows")]
                RawWindowHandle::Win32(h) => {
                    let hinstance = match raw_display_handle {
                        Some(RawDisplayHandle::Windows(d)) => d
                            .hinstance
                            .map(|p| p.get() as *mut _)
                            .unwrap_or(std::ptr::null_mut()),
                        _ => std::ptr::null_mut(),
                    };
                    let mut src = native::WGPUSurfaceSourceWindowsHWND {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceWindowsHWND,
                        },
                        hinstance,
                        hwnd: h.hwnd.get() as *mut _,
                    };
                    let c_desc = native::WGPUSurfaceDescriptor {
                        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                            &mut src.chain,
                        ),
                        label: conv::null_string_view(),
                    };
                    unsafe { wgpuInstanceCreateSurface(self.ptr, Some(&c_desc)) }
                }
                #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
                RawWindowHandle::Wayland(h) => {
                    let display = match raw_display_handle {
                        Some(RawDisplayHandle::Wayland(d)) => d.display.as_ptr(),
                        _ => std::ptr::null_mut(),
                    };
                    let mut src = native::WGPUSurfaceSourceWaylandSurface {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceWaylandSurface,
                        },
                        display,
                        surface: h.surface.as_ptr(),
                    };
                    let c_desc = native::WGPUSurfaceDescriptor {
                        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                            &mut src.chain,
                        ),
                        label: conv::null_string_view(),
                    };
                    unsafe { wgpuInstanceCreateSurface(self.ptr, Some(&c_desc)) }
                }
                #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
                RawWindowHandle::Xcb(h) => {
                    let connection = match raw_display_handle {
                        Some(RawDisplayHandle::Xcb(d)) => d
                            .connection
                            .map(|p| p.as_ptr())
                            .unwrap_or(std::ptr::null_mut()),
                        _ => std::ptr::null_mut(),
                    };
                    let mut src = native::WGPUSurfaceSourceXCBWindow {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceXCBWindow,
                        },
                        connection,
                        window: h.window,
                    };
                    let c_desc = native::WGPUSurfaceDescriptor {
                        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                            &mut src.chain,
                        ),
                        label: conv::null_string_view(),
                    };
                    unsafe { wgpuInstanceCreateSurface(self.ptr, Some(&c_desc)) }
                }
                #[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
                RawWindowHandle::Xlib(h) => {
                    let display = match raw_display_handle {
                        Some(RawDisplayHandle::Xlib(d)) => d
                            .display
                            .map(|p| p.as_ptr())
                            .unwrap_or(std::ptr::null_mut()),
                        _ => std::ptr::null_mut(),
                    };
                    let mut src = native::WGPUSurfaceSourceXlibWindow {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceXlibWindow,
                        },
                        display,
                        window: h.window,
                    };
                    let c_desc = native::WGPUSurfaceDescriptor {
                        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                            &mut src.chain,
                        ),
                        label: conv::null_string_view(),
                    };
                    unsafe { wgpuInstanceCreateSurface(self.ptr, Some(&c_desc)) }
                }
                _ => panic!("wgpu-c-backend: unsupported window handle type"),
            },
            _ => panic!("wgpu-c-backend: unsupported surface target type"),
        };

        if ptr.is_null() {
            panic!("wgpuInstanceCreateSurface returned null");
        }
        Ok(DispatchSurface::custom(surface::CSurface { ptr }))
    }

    fn request_adapter(
        &self,
        options: &wgpu::RequestAdapterOptions<'_, '_>,
    ) -> Pin<Box<dyn RequestAdapterFuture>> {
        struct Out {
            result: Option<Result<DispatchAdapter, wgpu::wgt::RequestAdapterError>>,
        }

        unsafe extern "C" fn cb(
            status: native::WGPURequestAdapterStatus,
            adapter: native::WGPUAdapter,
            _message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            out.result = Some(match status {
                native::WGPURequestAdapterStatus_Success => {
                    Ok(DispatchAdapter::custom(adapter::CAdapter { ptr: adapter }))
                }
                _ => Err(wgpu::wgt::RequestAdapterError::NotFound {
                    active_backends: wgpu::wgt::Backends::empty(),
                    requested_backends: wgpu::wgt::Backends::empty(),
                    supported_backends: wgpu::wgt::Backends::empty(),
                    no_fallback_backends: wgpu::wgt::Backends::empty(),
                    no_adapter_backends: wgpu::wgt::Backends::empty(),
                    incompatible_surface_backends: wgpu::wgt::Backends::empty(),
                }),
            });
        }

        let c_options = native::WGPURequestAdapterOptions {
            nextInChain: std::ptr::null_mut(),
            featureLevel: native::WGPUFeatureLevel_Core,
            powerPreference: conv::power_preference_to_native(options.power_preference),
            forceFallbackAdapter: options.force_fallback_adapter as u32,
            backendType: native::WGPUBackendType_Undefined,
            compatibleSurface: std::ptr::null_mut(),
        };

        let mut out = Out { result: None };
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(cb),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
            userdata2: std::ptr::null_mut(),
        };

        unsafe { wgpuInstanceRequestAdapter(self.ptr, Some(&c_options), callback_info) };
        let not_found = wgpu::wgt::RequestAdapterError::NotFound {
            active_backends: wgpu::wgt::Backends::empty(),
            requested_backends: wgpu::wgt::Backends::empty(),
            supported_backends: wgpu::wgt::Backends::empty(),
            no_fallback_backends: wgpu::wgt::Backends::empty(),
            no_adapter_backends: wgpu::wgt::Backends::empty(),
            incompatible_surface_backends: wgpu::wgt::Backends::empty(),
        };
        Box::pin(future::ready(out.result.unwrap_or(Err(not_found))))
    }

    fn poll_all_devices(&self, _force_wait: bool) -> bool {
        // wgpu-native has no equivalent poll_all_devices.
        unimplemented!("wgpu-native does not expose poll_all_devices")
    }

    fn enumerate_adapters(&self, backends: wgpu::Backends) -> Pin<Box<dyn EnumerateAdapterFuture>> {
        let options = native::WGPUInstanceEnumerateAdapterOptions {
            backends: conv::backends_to_native(backends),
            nextInChain: std::ptr::null(),
        };

        let adapters = unsafe {
            let count =
                wgpuInstanceEnumerateAdapters(self.ptr, Some(&options), std::ptr::null_mut());

            let mut out: Vec<native::WGPUAdapter> = vec![std::ptr::null_mut(); count];
            wgpuInstanceEnumerateAdapters(self.ptr, Some(&options), out.as_mut_ptr());

            out.into_iter()
                .map(|ptr| DispatchAdapter::custom(CAdapter { ptr }))
                .collect::<Vec<_>>()
        };

        Box::pin(future::ready(adapters))
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        // wgpu-native has no WGSL language features query.
        unimplemented!("wgpu-native does not expose WGSL language features")
    }
}
