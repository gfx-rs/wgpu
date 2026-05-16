use std::future;
use std::pin::Pin;

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;
use crate::device::{CDevice, CQueue};

pub(crate) fn adapter_info(info: &native::WGPUAdapterInfo) -> wgpu::AdapterInfo {
    let device_type = conv::map_device_type_from_native(info.adapterType);
    let backend = conv::map_backend_from_native(info.backendType);
    wgpu::AdapterInfo {
        name: unsafe { conv::string_view_to_string(info.device) },
        vendor: info.vendorID,
        device: info.deviceID,
        device_type,
        device_pci_bus_id: String::new(),
        driver: unsafe { conv::string_view_to_string(info.vendor) },
        driver_info: unsafe { conv::string_view_to_string(info.description) },
        backend,
        subgroup_min_size: wgpu::wgt::MINIMUM_SUBGROUP_MIN_SIZE,
        subgroup_max_size: wgpu::wgt::MAXIMUM_SUBGROUP_MAX_SIZE,
        transient_saves_memory: false,
        limit_bucket: None,
    }
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
                callback: None,
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
        };

        struct Out {
            device: native::WGPUDevice,
            status: native::WGPURequestDeviceStatus,
        }
        let mut out = Out {
            device: std::ptr::null_mut(),
            status: native::WGPURequestDeviceStatus_Error,
        };

        unsafe extern "C" fn callback(
            status: native::WGPURequestDeviceStatus,
            device: native::WGPUDevice,
            _message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            out.status = status;
            out.device = device;
        }

        let callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(callback),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
            userdata2: std::ptr::null_mut(),
        };

        // Capture adapter info before creating device (needed for device.adapter_info()).
        let info = unsafe {
            let mut raw: native::WGPUAdapterInfo = std::mem::zeroed();
            wgpuAdapterGetInfo(self.ptr, Some(&mut raw));
            let parsed = adapter_info(&raw);
            wgpuAdapterInfoFreeMembers(raw);
            parsed
        };

        unsafe { wgpuAdapterRequestDevice(self.ptr, Some(&c_desc), callback_info) };

        let result =
            if out.status == native::WGPURequestDeviceStatus_Success && !out.device.is_null() {
                let queue_ptr = unsafe { wgpuDeviceGetQueue(out.device) };
                Ok((
                    DispatchDevice::custom(CDevice {
                        ptr: out.device,
                        info,
                    }),
                    DispatchQueue::custom(CQueue { ptr: queue_ptr }),
                ))
            } else {
                panic!("wgpu-native: request_device failed")
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
        // wgpu-native has no downlevel capabilities query.
        unimplemented!("wgpu-native does not expose downlevel capabilities")
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        let mut raw: native::WGPUAdapterInfo = unsafe { std::mem::zeroed() };
        unsafe { wgpuAdapterGetInfo(self.ptr, Some(&mut raw)) };
        let info = adapter_info(&raw);
        unsafe { wgpuAdapterInfoFreeMembers(raw) };
        info
    }

    fn get_texture_format_features(
        &self,
        _format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        // wgpu-native has no per-format feature query on the adapter.
        unimplemented!("wgpu-native does not expose texture format feature queries")
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
