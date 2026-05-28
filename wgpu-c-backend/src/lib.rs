mod adapter;
mod command;
mod conv;
mod device;
mod pass;
mod resource;
mod surface;

use std::future;
use std::pin::Pin;

// ── Panic propagation for extern "C" callbacks ────────────────────────────────
//
// Rust panics must not cross `extern "C"` boundaries (UB). When a user-supplied
// Rust closure is called from within one of our `extern "C"` C-API callbacks,
// we catch any panic with `catch_unwind` and store it here. The caller then
// checks `resume_callback_panic()` after the C function returns to re-raise it
// on the Rust side where it can propagate normally.
//
// We use a global `Mutex` rather than `thread_local!` so that panics from
// callbacks that fire spontaneously on wgpu-native's background threads
// (WGPUCallbackMode_AllowSpontaneous) are visible when `resume_callback_panic`
// is called on the test/calling thread. Only the first panic is kept; subsequent
// ones are silently dropped (matching the previous per-thread behaviour).
pub(crate) static CALLBACK_PANIC: std::sync::Mutex<
    Option<Box<dyn std::any::Any + Send + 'static>>,
> = std::sync::Mutex::new(None);

pub(crate) fn catch_callback_panic<F: FnOnce()>(f: F) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        let mut guard = CALLBACK_PANIC.lock().unwrap();
        if guard.is_none() {
            *guard = Some(payload);
        }
    }
}

pub(crate) fn resume_callback_panic() {
    if let Some(payload) = CALLBACK_PANIC.lock().unwrap().take() {
        std::panic::resume_unwind(payload);
    }
}

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

// Backends that wgpu-native actually implements.
const WGPU_NATIVE_BACKENDS: wgpu::Backends = wgpu::Backends::VULKAN
    .union(wgpu::Backends::METAL)
    .union(wgpu::Backends::DX12)
    .union(wgpu::Backends::GL);

#[expect(clippy::missing_safety_doc)]
#[expect(improper_ctypes_definitions)]
#[expect(clippy::result_large_err)]
#[no_mangle]
pub unsafe extern "C" fn instance_factory(
    desc: InstanceDescriptor,
) -> Result<wgpu::Instance, InstanceDescriptor> {
    // Pass through to wgpu-core's built-in factory when the requested backends
    // don't include anything wgpu-native can handle (e.g. Backends::empty(),
    // Backends::NOOP, Backends::BROWSER_WEBGPU). wgpu-core will generate the
    // correct "not requested" / "not compiled in" error messages.
    if desc.backends.intersection(WGPU_NATIVE_BACKENDS).is_empty() {
        return Err(desc);
    }
    Ok(wgpu::Instance::from_custom(CInstance::new(desc)))
}

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
                    // ns_view is an NSView*, not a CAMetalLayer*. Use raw_window_metal to
                    // install/retrieve a CAMetalLayer on the view before handing it to wgpu-native.
                    let layer = unsafe { raw_window_metal::Layer::from_ns_view(h.ns_view) };
                    let mut src = native::WGPUSurfaceSourceMetalLayer {
                        chain: native::WGPUChainedStruct {
                            next: std::ptr::null_mut(),
                            sType: native::WGPUSType_SurfaceSourceMetalLayer,
                        },
                        layer: layer.as_ptr().as_ptr().cast(),
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

        // Extract the raw WGPUSurface pointer from the compatible_surface if provided.
        // Surface::as_custom returns None if the surface was not created by this backend,
        // in which case we fall back to null (no surface constraint).
        let compatible_surface_ptr: native::WGPUSurface = options
            .compatible_surface
            .and_then(|s| s.as_custom::<surface::CSurface>())
            .map(|cs| cs.ptr)
            .unwrap_or(std::ptr::null_mut());

        let c_options = native::WGPURequestAdapterOptions {
            nextInChain: std::ptr::null_mut(),
            featureLevel: native::WGPUFeatureLevel_Undefined,
            powerPreference: conv::power_preference_to_native(options.power_preference),
            forceFallbackAdapter: options.force_fallback_adapter as u32,
            backendType: native::WGPUBackendType_Undefined,
            compatibleSurface: compatible_surface_ptr,
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

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        unsafe { wgpuInstancePollAllDevices(self.ptr, force_wait) }
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
        let bits = wgpu_native::wgpuGetWgslLanguageFeatures();
        let mut out = wgpu::WgslLanguageFeatures::empty();
        if bits & wgpu_native::native::WGPUWgslLanguageFeatures_ReadOnlyAndReadWriteStorageTextures
            != 0
        {
            out |= wgpu::WgslLanguageFeatures::ReadOnlyAndReadWriteStorageTextures;
        }
        if bits & wgpu_native::native::WGPUWgslLanguageFeatures_Packed4x8IntegerDotProduct != 0 {
            out |= wgpu::WgslLanguageFeatures::Packed4x8IntegerDotProduct;
        }
        if bits & wgpu_native::native::WGPUWgslLanguageFeatures_PointerCompositeAccess != 0 {
            out |= wgpu::WgslLanguageFeatures::PointerCompositeAccess;
        }
        out
    }
}
