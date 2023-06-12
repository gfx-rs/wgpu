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
