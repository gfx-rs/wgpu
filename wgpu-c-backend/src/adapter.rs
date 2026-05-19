use std::future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;
use crate::device::{CDevice, CQueue, ErrorHandler};

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

        // Build the limits struct.
        let c_limits = conv::limits_to_native(&desc.required_limits);

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
                handler(error);
            }
            // If no handler set, silently ignore (don't abort like wgpu-native's default).
        }

        let error_handler: ErrorHandler = Arc::new(Mutex::new(None));

        let c_desc = native::WGPUDeviceDescriptor {
            nextInChain: std::ptr::null_mut(),
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
                callback: None,
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
            uncapturedErrorCallbackInfo: native::WGPUUncapturedErrorCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                callback: Some(uncaptured_error_cb),
                // SAFETY: error_handler outlives the device (stored in CDevice).
                // wgpu-native will not call the callback after wgpuDeviceRelease.
                userdata1: Arc::as_ptr(&error_handler) as *mut _,
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
                Ok((
                    DispatchDevice::custom(CDevice {
                        ptr: out.device,
                        info,
                        error_handler,
                    }),
                    DispatchQueue::custom(CQueue { ptr: queue_ptr }),
                ))
            } else {
                Err(wgpu::RequestDeviceError::from_message(format!(
                    "wgpu-native: request_device failed (status={}, message={})",
                    out.status, out.message
                )))
            };

        Box::pin(future::ready(result))
    }

    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        // wgpu-native has no wgpuAdapterIsSurfaceSupported equivalent.
        unimplemented!("wgpu-native does not expose adapter surface support query")
    }

    fn features(&self) -> wgpu::Features {
        let mut supported: native::WGPUSupportedFeatures = unsafe { std::mem::zeroed() };
        unsafe { wgpuAdapterGetFeatures(self.ptr, Some(&mut supported)) };
        let result = conv::map_supported_features(&supported);
        unsafe { wgpuSupportedFeaturesFreeMembers(supported) };
        result
    }

    fn limits(&self) -> wgpu::Limits {
        let mut limits: native::WGPULimits = unsafe { std::mem::zeroed() };
        unsafe { wgpuAdapterGetLimits(self.ptr, Some(&mut limits)) };
        conv::map_limits(&limits)
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        get_adapter_info(self.ptr)
    }

    fn get_texture_format_features(
        &self,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        // wgpu-native has no per-format feature query, so fall back to the
        // WebGPU-guaranteed minimums, conditioned on the adapter's actual features.
        format.guaranteed_format_features(self.features())
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        // wgpu-native has no presentation timestamp query.
        unimplemented!("wgpu-native does not expose presentation timestamps")
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        // wgpu-native has no cooperative matrix properties query.
        unimplemented!("wgpu-native does not expose cooperative matrix properties")
    }
}
