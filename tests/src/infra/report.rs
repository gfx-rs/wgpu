use std::collections::HashMap;

use serde::Deserialize;
use wgpu::{
    AdapterInfo, DownlevelCapabilities, Features, Limits, TextureFormat, TextureFormatFeatures,
};

#[derive(Deserialize)]
pub struct GpuReport {
    pub devices: Vec<AdapterReport>,
}

impl GpuReport {
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    pub fn from_json(file: &str) -> serde_json::Result<Self> {
        serde_json::from_str(file)
    }
}

#[derive(Deserialize)]
pub struct AdapterReport {
    pub info: AdapterInfo,
    pub features: Features,
    pub limits: Limits,
    pub downlevel_caps: DownlevelCapabilities,
    pub texture_format_features: HashMap<TextureFormat, TextureFormatFeatures>,
}

impl AdapterReport {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
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
