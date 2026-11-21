use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wgpu::{
    AdapterInfo, DownlevelCapabilities, Features, Limits, TextureFormat, TextureFormatFeatures,
};

/// Report specifying the capabilities of the GPUs on the system.
///
/// Must be synchronized with the definition on wgpu-info/src/report.rs.
#[derive(Serialize, Deserialize)]
pub(crate) struct GpuReport {
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    pub devices: Vec<AdapterReport>,
}

impl GpuReport {
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates a new GpuReport with a single noop adapter.
    pub(crate) fn noop_only() -> Self {
        GpuReport {
            devices: vec![AdapterReport {
                info: wgpu::hal::noop::adapter_info(),
                features: Features::all(),
                limits: wgpu::hal::noop::CAPABILITIES.limits,
                downlevel_caps: wgpu::hal::noop::CAPABILITIES.downlevel,
                texture_format_features: HashMap::new(), // todo
                vulkan_api_version: None,
            }],
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    pub(crate) fn from_json(file: &str) -> serde_json::Result<Self> {
        profiling::scope!("Parsing .gpuconfig");
        serde_json::from_str(file)
    }
}

/// A single report of the capabilities of an Adapter.
///
/// Must be synchronized with the definition on wgpu-info/src/report.rs.
#[derive(Serialize, Deserialize, Clone)]
pub struct AdapterReport {
    pub info: AdapterInfo,
    pub features: Features,
    pub limits: Limits,
    pub downlevel_caps: DownlevelCapabilities,
    pub texture_format_features: HashMap<TextureFormat, TextureFormatFeatures>,
    /// The Vulkan API version the driver reports. Only set for Vulkan adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_api_version: Option<[u32; 3]>,
}

impl AdapterReport {
    pub(crate) fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        let downlevel_caps = adapter.get_downlevel_capabilities();

        Self {
            info,
            features,
            limits,
            downlevel_caps,
            texture_format_features: HashMap::new(), // todo
            vulkan_api_version: vulkan_api_version(adapter),
        }
    }
}

/// The Vulkan API version the adapter's driver reports, as `[major, minor, patch]`.
fn vulkan_api_version(adapter: &wgpu::Adapter) -> Option<[u32; 3]> {
    #[cfg(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "macos"
    ))]
    {
        use wgpu::hal::vulkan::Api;
        // SAFETY: the hal adapter is only used to read its properties.
        let adapter = unsafe { adapter.as_hal::<Api>() }?;
        let version = adapter
            .physical_device_capabilities()
            .properties()
            .api_version;
        // Decoded as `VK_API_VERSION_MAJOR`, `_MINOR` and `_PATCH` do.
        Some([
            (version >> 22) & 0x7f,
            (version >> 12) & 0x3ff,
            version & 0xfff,
        ])
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "macos"
    )))]
    {
        let _ = adapter;
        None
    }
}
