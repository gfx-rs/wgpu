use std::collections::HashMap;

use serde::Deserialize;
use wgpu::{
    AdapterInfo, DownlevelCapabilities, Features, Limits, TextureFormat, TextureFormatFeatures,
};

/// Report specifying the capabilities of the GPUs on the system.
///
/// Must be synchronized with the definition on wgpu-info/src/report.rs.
#[derive(Deserialize)]
pub(crate) struct GpuReport {
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    pub devices: Vec<AdapterReport>,
}

impl GpuReport {
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    pub(crate) fn from_json(file: &str) -> serde_json::Result<Self> {
        profiling::scope!("Parsing .gpuconfig");
        serde_json::from_str(file)
    }
}

/// A single report of the capabilities of an Adapter.
///
/// Must be synchronized with the definition on wgpu-info/src/report.rs.
#[derive(Deserialize, Clone)]
pub struct AdapterReport {
    pub info: AdapterInfo,
    pub features: Features,
    pub limits: Limits,
    pub downlevel_caps: DownlevelCapabilities,
    #[allow(unused)]
    pub texture_format_features: HashMap<TextureFormat, TextureFormatFeatures>,
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
        }
    }
}
