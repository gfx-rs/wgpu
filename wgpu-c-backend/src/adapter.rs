use std::future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex};

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;
use crate::device::{CDevice, CQueue, DeviceLostHandler, ErrorHandler};

pub(crate) fn adapter_info_with_extras(
    info: &native::WGPUAdapterInfo,
    extras: &native::WGPUAdapterInfoExtras,
) -> wgpu::AdapterInfo {
    let device_type = conv::map_device_type_from_native(info.adapterType);
    let backend = conv::map_backend_from_native(info.backendType);
    wgpu::AdapterInfo {
        name: unsafe { conv::string_view_to_string(info.device) },
        vendor: info.vendorID,
        device: info.deviceID,
        device_type,
        device_pci_bus_id: unsafe { conv::string_view_to_string(extras.devicePciBusId) },
        driver: unsafe { conv::string_view_to_string(info.vendor) },
        driver_info: unsafe { conv::string_view_to_string(info.description) },
        backend,
        subgroup_min_size: info.subgroupMinSize,
        subgroup_max_size: info.subgroupMaxSize,
        transient_saves_memory: extras.transientSavesMemory != 0,
        limit_bucket: None,
    }
}

pub(crate) fn get_adapter_info(adapter: native::WGPUAdapter) -> wgpu::AdapterInfo {
    let mut extras = native::WGPUAdapterInfoExtras {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_AdapterInfoExtras,
        },
        transientSavesMemory: 0,
        devicePciBusId: conv::null_string_view(),
    };
    let mut raw = native::WGPUAdapterInfo {
        nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut extras.chain),
        ..unsafe { std::mem::zeroed() }
    };
    unsafe { wgpuAdapterGetInfo(adapter, Some(&mut raw)) };
    let info = adapter_info_with_extras(&raw, &extras);
    unsafe { wgpuAdapterInfoFreeMembers(raw) };
    info
}

// ── CAdapter ──────────────────────────────────────────────────────────────────

pub struct CAdapter {
    pub(crate) ptr: native::WGPUAdapter,
}

impl std::fmt::Debug for CAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CAdapter").field("ptr", &self.ptr).finish()
    }
}

unsafe impl Send for CAdapter {}
unsafe impl Sync for CAdapter {}

impl Drop for CAdapter {
    fn drop(&mut self) {
        unsafe { wgpuAdapterRelease(self.ptr) };
    }
}

impl AdapterInterface for CAdapter {
    fn request_device(
        &self,
        desc: &wgpu::DeviceDescriptor<'_>,
    ) -> Pin<Box<dyn RequestDeviceFuture>> {
        // Build the feature list from required_features.
        let mut required_features = conv::features_to_native(desc.required_features);

        // Build the limits structs. The standard WGPULimits covers WebGPU core limits;
        // WGPUNativeLimits covers extended wgpu-native limits (BLAS, mesh shaders, etc.).
        let mut c_native_limits = conv::native_limits_from_wgpu(&desc.required_limits);
        let mut c_limits = conv::limits_to_native(&desc.required_limits);
        c_limits.nextInChain =
            std::ptr::from_mut::<native::WGPUChainedStruct>(&mut c_native_limits.chain);

        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        // Uncaptured error callback: delegates to the handler registered via
        // on_uncaptured_error(), or silently ignores if none is set.
        unsafe extern "C" fn uncaptured_error_cb(
            _device: *const native::WGPUDevice,
            type_: native::WGPUErrorType,
            message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let handler_arc = unsafe { &*(userdata1 as *const ErrorHandler) };
            let guard = handler_arc.lock().unwrap();
            let msg = unsafe { crate::conv::string_view_to_string(message) };
            if let Some(handler) = guard.as_ref() {
                let error = match type_ {
                    native::WGPUErrorType_Validation => wgpu::Error::Validation {
                        source: Box::new(std::io::Error::other(msg.clone())),
                        description: msg,
                    },
                    native::WGPUErrorType_OutOfMemory => wgpu::Error::OutOfMemory {
                        source: Box::new(std::io::Error::other(msg)),
                    },
                    _ => wgpu::Error::Internal {
                        source: Box::new(std::io::Error::other(msg.clone())),
                        description: msg,
                    },
                };
                let handler = Arc::clone(handler);
                drop(guard);
                crate::catch_callback_panic(|| handler(error));
            }
            // If no handler set, silently ignore (don't abort like wgpu-native's default).
        }

        // Device lost callback: delegates to the handler registered via set_device_lost_callback().
        unsafe extern "C" fn device_lost_cb(
            _device: *const native::WGPUDevice,
            reason: native::WGPUDeviceLostReason,
            message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let handler = unsafe { &*(userdata1 as *const DeviceLostHandler) };
            let callback = handler.lock().unwrap().take();
            if let Some(callback) = callback {
                let reason_wgpu = match reason {
                    native::WGPUDeviceLostReason_Destroyed => wgpu::DeviceLostReason::Destroyed,
                    _ => wgpu::DeviceLostReason::Unknown,
                };
                let msg = unsafe { crate::conv::string_view_to_string(message) };
                crate::catch_callback_panic(|| callback(reason_wgpu, msg));
            }
        }

        // Box gives a stable heap address for the Arc itself. We pass
        // a *const Arc<Mutex<...>> (= *const ErrorHandler) as userdata1 so
        // the callback can safely reconstruct &ErrorHandler via a pointer cast.
        // Arc::as_ptr would return *const Mutex<...> (the inner T), not a pointer
        // to the Arc struct — casting that to *const Arc would be UB / SIGSEGV.
        let error_handler: Box<ErrorHandler> = Box::new(Arc::new(Mutex::new(None)));
        let handler_ptr = error_handler.as_ref() as *const ErrorHandler;

        let device_lost_handler: Box<DeviceLostHandler> = Box::new(Mutex::new(None));
        let device_lost_ptr = device_lost_handler.as_ref() as *const DeviceLostHandler;

        let (memory_hints, mem_min, mem_max) =
            conv::memory_hints_to_native(&desc.memory_hints);
        let mut device_extras = native::WGPUDeviceDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_DeviceDescriptorExtras,
            },
            memoryHints: memory_hints,
            suballocatedDeviceMemoryBlockSizeMin: mem_min,
            suballocatedDeviceMemoryBlockSizeMax: mem_max,
            experimentalFeaturesEnabled: desc.experimental_features.is_enabled() as _,
        };
        let c_desc = native::WGPUDeviceDescriptor {
            nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut device_extras.chain),
            label: label_sv,
            requiredFeatureCount: required_features.len(),
            requiredFeatures: required_features.as_mut_ptr(),
            requiredLimits: &c_limits,
            defaultQueue: native::WGPUQueueDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: conv::null_string_view(),
            },
            deviceLostCallbackInfo: native::WGPUDeviceLostCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowSpontaneous,
                callback: Some(device_lost_cb),
                // SAFETY: device_lost_ptr points into the Box heap allocation, which is
                // stored in CDevice and outlives the device.
                userdata1: device_lost_ptr as *mut _,
                userdata2: std::ptr::null_mut(),
            },
            uncapturedErrorCallbackInfo: native::WGPUUncapturedErrorCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                callback: Some(uncaptured_error_cb),
                // SAFETY: handler_ptr points into the Box heap allocation, which is
                // stored in CDevice and outlives the device. wgpu-native will not call
                // the callback after wgpuDeviceRelease.
                userdata1: handler_ptr as *mut _,
                userdata2: std::ptr::null_mut(),
            },
        };

        struct Out {
            device: native::WGPUDevice,
            status: native::WGPURequestDeviceStatus,
            message: String,
        }
        let mut out = Out {
            device: std::ptr::null_mut(),
            status: native::WGPURequestDeviceStatus_Error,
            message: String::new(),
        };

        unsafe extern "C" fn callback(
            status: native::WGPURequestDeviceStatus,
            device: native::WGPUDevice,
            message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            out.status = status;
            out.device = device;
            out.message = unsafe { crate::conv::string_view_to_string(message) };
        }

        let callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(callback),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
            userdata2: std::ptr::null_mut(),
        };

        // Capture adapter info before creating device (needed for device.adapter_info()).
        let info = get_adapter_info(self.ptr);

        unsafe { wgpuAdapterRequestDevice(self.ptr, Some(&c_desc), callback_info) };

        let result =
            if out.status == native::WGPURequestDeviceStatus_Success && !out.device.is_null() {
                let queue_ptr = unsafe { wgpuDeviceGetQueue(out.device) };
                let error_scope_depth = Arc::new(AtomicU32::new(0));
                let queue_dropped = Arc::new(AtomicBool::new(false));
                Ok((
                    DispatchDevice::custom(CDevice {
                        ptr: out.device,
                        info,
                        error_handler,
                        device_lost_handler,
                        error_scope_depth: Arc::clone(&error_scope_depth),
                        queue_dropped: Arc::clone(&queue_dropped),
                    }),
                    DispatchQueue::custom(CQueue {
                        ptr: queue_ptr,
                        device_ptr: out.device,
                        error_scope_depth,
                        queue_dropped,
                    }),
                ))
            } else {
                Err(wgpu::RequestDeviceError::from_message(format!(
                    "wgpu-native: request_device failed (status={}, message={})",
                    out.status, out.message
                )))
            };

        Box::pin(future::ready(result))
    }

    fn is_surface_supported(&self, surface: &DispatchSurface) -> bool {
        let surface_ptr = surface.as_custom::<crate::surface::CSurface>().unwrap().ptr;
        unsafe { wgpuAdapterIsSurfaceSupported(self.ptr, surface_ptr) != 0 }
    }

    fn features(&self) -> wgpu::Features {
        let mut supported: native::WGPUSupportedFeatures = unsafe { std::mem::zeroed() };
        unsafe { wgpuAdapterGetFeatures(self.ptr, Some(&mut supported)) };
        let result = conv::map_supported_features(&supported);
        unsafe { wgpuSupportedFeaturesFreeMembers(supported) };
        result
    }

    fn limits(&self) -> wgpu::Limits {
        let mut native_limits: native::WGPUNativeLimits = unsafe { std::mem::zeroed() };
        native_limits.chain = native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_NativeLimits,
        };
        let mut limits: native::WGPULimits = unsafe { std::mem::zeroed() };
        limits.nextInChain = std::ptr::from_mut::<native::WGPUChainedStruct>(&mut native_limits.chain);
        unsafe { wgpuAdapterGetLimits(self.ptr, Some(&mut limits)) };
        conv::map_limits(&limits, Some(&native_limits))
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        let c = unsafe { wgpuAdapterGetDownlevelCapabilities(self.ptr) };
        let mut flags = wgpu::DownlevelFlags::empty();
        macro_rules! flag {
            ($native:ident => $wgpu:ident) => {
                if c.flags & native::$native != 0 {
                    flags |= wgpu::DownlevelFlags::$wgpu;
                }
            };
        }
        flag!(WGPUDownlevelFlags_ComputeShaders => COMPUTE_SHADERS);
        flag!(WGPUDownlevelFlags_FragmentWritableStorage => FRAGMENT_WRITABLE_STORAGE);
        flag!(WGPUDownlevelFlags_IndirectExecution => INDIRECT_EXECUTION);
        flag!(WGPUDownlevelFlags_BaseVertex => BASE_VERTEX);
        flag!(WGPUDownlevelFlags_ReadOnlyDepthStencil => READ_ONLY_DEPTH_STENCIL);
        flag!(WGPUDownlevelFlags_NonPowerOfTwoMipmappedTextures => NON_POWER_OF_TWO_MIPMAPPED_TEXTURES);
        flag!(WGPUDownlevelFlags_CubeArrayTextures => CUBE_ARRAY_TEXTURES);
        flag!(WGPUDownlevelFlags_ComparisonSamplers => COMPARISON_SAMPLERS);
        flag!(WGPUDownlevelFlags_IndependentBlend => INDEPENDENT_BLEND);
        flag!(WGPUDownlevelFlags_VertexStorage => VERTEX_STORAGE);
        flag!(WGPUDownlevelFlags_AnisotropicFiltering => ANISOTROPIC_FILTERING);
        flag!(WGPUDownlevelFlags_FragmentStorage => FRAGMENT_STORAGE);
        flag!(WGPUDownlevelFlags_MultisampledShading => MULTISAMPLED_SHADING);
        flag!(WGPUDownlevelFlags_DepthTextureAndBufferCopies => DEPTH_TEXTURE_AND_BUFFER_COPIES);
        flag!(WGPUDownlevelFlags_WebGpuTextureFormatSupport => WEBGPU_TEXTURE_FORMAT_SUPPORT);
        flag!(WGPUDownlevelFlags_BufferBindingsNot16ByteAligned => BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED);
        flag!(WGPUDownlevelFlags_UnrestrictedIndexBuffer => UNRESTRICTED_INDEX_BUFFER);
        flag!(WGPUDownlevelFlags_FullDrawIndexUint32 => FULL_DRAW_INDEX_UINT32);
        flag!(WGPUDownlevelFlags_DepthBiasClamp => DEPTH_BIAS_CLAMP);
        flag!(WGPUDownlevelFlags_ViewFormats => VIEW_FORMATS);
        flag!(WGPUDownlevelFlags_UnrestrictedExternalTextureCopies => UNRESTRICTED_EXTERNAL_TEXTURE_COPIES);
        flag!(WGPUDownlevelFlags_SurfaceViewFormats => SURFACE_VIEW_FORMATS);
        flag!(WGPUDownlevelFlags_NonblockingQueryResolve => NONBLOCKING_QUERY_RESOLVE);
        flag!(WGPUDownlevelFlags_ShaderF16InF32 => SHADER_F16_IN_F32);
        flag!(WGPUDownlevelFlags_Msl21 => MSL2_1);
        let shader_model = match c.shaderModel {
            native::WGPUShaderModel_Sm2 => wgpu::ShaderModel::Sm2,
            native::WGPUShaderModel_Sm4 => wgpu::ShaderModel::Sm4,
            _ => wgpu::ShaderModel::Sm5,
        };
        wgpu::DownlevelCapabilities { flags, limits: wgpu::DownlevelLimits {}, shader_model }
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        get_adapter_info(self.ptr)
    }

    fn get_texture_format_features(
        &self,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        let native_fmt = conv::texture_format_to_native(format);
        if native_fmt == native::WGPUTextureFormat_Undefined {
            return format.guaranteed_format_features(self.features());
        }
        let mut caps = native::WGPUNativeTextureFormatCapabilities {
            allowedUsages: 0,
            flags: 0,
        };
        let status = unsafe {
            wgpuAdapterGetTextureFormatCapabilities(self.ptr, native_fmt, Some(&mut caps))
        };
        if status != native::WGPUStatus_Success {
            return format.guaranteed_format_features(self.features());
        }
        conv::map_texture_format_capabilities(&caps)
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        let ts = unsafe { wgpuAdapterGetPresentationTimestamp(self.ptr) };
        wgpu::PresentationTimestamp(ts.nanoseconds as u128)
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        let count =
            unsafe { wgpuAdapterGetCooperativeMatrixProperties(self.ptr, std::ptr::null_mut(), 0) };
        if count == 0 {
            return Vec::new();
        }
        let mut c_props = vec![
            native::WGPUCooperativeMatrixProperties {
                mSize: 0,
                nSize: 0,
                kSize: 0,
                abType: native::WGPUNativeCooperativeScalarType_F32,
                crType: native::WGPUNativeCooperativeScalarType_F32,
                saturatingAccumulation: 0,
            };
            count
        ];
        unsafe {
            wgpuAdapterGetCooperativeMatrixProperties(
                self.ptr,
                c_props.as_mut_ptr(),
                c_props.len(),
            )
        };
        c_props
            .into_iter()
            .map(|p| wgpu::wgt::CooperativeMatrixProperties {
                m_size: p.mSize,
                n_size: p.nSize,
                k_size: p.kSize,
                ab_type: conv::cooperative_scalar_type_from_native(p.abType),
                cr_type: conv::cooperative_scalar_type_from_native(p.crType),
                saturating_accumulation: p.saturatingAccumulation != 0,
            })
            .collect()
    }
}
