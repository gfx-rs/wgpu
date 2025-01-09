use std::{collections::HashMap, io};

use serde::{Deserialize, Serialize};
use wgpu::{
    AdapterInfo, DownlevelCapabilities, Features, Limits, TextureFormat, TextureFormatFeatures,
};

use crate::texture;

/// Report specifying the capabilities of the GPUs on the system.
///
/// Must be synchronized with the definition on tests/src/report.rs.
#[derive(Deserialize, Serialize)]
pub struct GpuReport {
    pub devices: Vec<AdapterReport>,
}

impl GpuReport {
    pub fn generate() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or_default(),
            flags: wgpu::InstanceFlags::debugging().with_env(),
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env()
                .unwrap_or(wgpu::Dx12Compiler::StaticDxc),
            gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        let mut devices = Vec::with_capacity(adapters.len());
        for adapter in adapters {
            let features = adapter.features();
            let limits = adapter.limits();
            let downlevel_caps = adapter.get_downlevel_capabilities();
            let texture_format_features = texture::TEXTURE_FORMAT_LIST
                .into_iter()
                .map(|format| (format, adapter.get_texture_format_features(format)))
                .collect();

            devices.push(AdapterReport {
                info: adapter.get_info(),
                features,
                limits,
                downlevel_caps,
                texture_format_features,
            });
        }

        Self { devices }
    }

    pub fn from_json(file: &str) -> serde_json::Result<Self> {
        serde_json::from_str(file)
    }

    pub fn into_json(self, output: impl io::Write) -> serde_json::Result<()> {
        serde_json::to_writer_pretty(output, &self)
    }
}

/// A single report of the capabilities of an Adapter.
///
/// Must be synchronized with the definition on tests/src/report.rs.
#[derive(Deserialize, Serialize)]
pub struct AdapterReport {
    pub info: AdapterInfo,
    pub features: Features,
    pub limits: Limits,
    pub downlevel_caps: DownlevelCapabilities,
    pub texture_format_features: HashMap<TextureFormat, TextureFormatFeatures>,
}
