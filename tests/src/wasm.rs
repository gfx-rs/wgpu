#![cfg(target_arch = "wasm32")]

use crate::{
    execute_test,
    init::init_logger,
    initialize_html_canvas,
    report::{AdapterReport, GpuReport},
    GpuTestInitializer,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use wgpu::Backends;

// @todo avoid copying this from wgpu-info
#[rustfmt::skip]
pub const TEXTURE_FORMAT_LIST: [wgpu::TextureFormat; 118] = [
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R8Snorm,
    wgpu::TextureFormat::R8Uint,
    wgpu::TextureFormat::R8Sint,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::R16Sint,
    wgpu::TextureFormat::R16Unorm,
    wgpu::TextureFormat::R16Snorm,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::Rg8Unorm,
    wgpu::TextureFormat::Rg8Snorm,
    wgpu::TextureFormat::Rg8Uint,
    wgpu::TextureFormat::Rg8Sint,
    wgpu::TextureFormat::R32Uint,
    wgpu::TextureFormat::R32Sint,
    wgpu::TextureFormat::R32Float,
    wgpu::TextureFormat::Rg16Uint,
    wgpu::TextureFormat::Rg16Sint,
    wgpu::TextureFormat::Rg16Unorm,
    wgpu::TextureFormat::Rg16Snorm,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Snorm,
    wgpu::TextureFormat::Rgba8Uint,
    wgpu::TextureFormat::Rgba8Sint,
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgb9e5Ufloat,
    wgpu::TextureFormat::Rgb10a2Uint,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rg11b10Ufloat,
    wgpu::TextureFormat::R64Uint,
    wgpu::TextureFormat::Rg32Uint,
    wgpu::TextureFormat::Rg32Sint,
    wgpu::TextureFormat::Rg32Float,
    wgpu::TextureFormat::Rgba16Uint,
    wgpu::TextureFormat::Rgba16Sint,
    wgpu::TextureFormat::Rgba16Unorm,
    wgpu::TextureFormat::Rgba16Snorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Uint,
    wgpu::TextureFormat::Rgba32Sint,
    wgpu::TextureFormat::Rgba32Float,
    wgpu::TextureFormat::Stencil8,
    wgpu::TextureFormat::Depth16Unorm,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth24PlusStencil8,
    wgpu::TextureFormat::Depth32Float,
    wgpu::TextureFormat::Depth32FloatStencil8,
    wgpu::TextureFormat::NV12,
    wgpu::TextureFormat::P010,
    wgpu::TextureFormat::Bc1RgbaUnorm,
    wgpu::TextureFormat::Bc1RgbaUnormSrgb,
    wgpu::TextureFormat::Bc2RgbaUnorm,
    wgpu::TextureFormat::Bc2RgbaUnormSrgb,
    wgpu::TextureFormat::Bc3RgbaUnorm,
    wgpu::TextureFormat::Bc3RgbaUnormSrgb,
    wgpu::TextureFormat::Bc4RUnorm,
    wgpu::TextureFormat::Bc4RSnorm,
    wgpu::TextureFormat::Bc5RgUnorm,
    wgpu::TextureFormat::Bc5RgSnorm,
    wgpu::TextureFormat::Bc6hRgbUfloat,
    wgpu::TextureFormat::Bc6hRgbFloat,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Etc2Rgb8Unorm,
    wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
    wgpu::TextureFormat::Etc2Rgb8A1Unorm,
    wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
    wgpu::TextureFormat::Etc2Rgba8Unorm,
    wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
    wgpu::TextureFormat::EacR11Unorm,
    wgpu::TextureFormat::EacR11Snorm,
    wgpu::TextureFormat::EacRg11Unorm,
    wgpu::TextureFormat::EacRg11Snorm,
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Hdr },
];

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = "
  export function test_success() {
    window.sessionStorage.test_success = `true`;
  }

  export function test_failure(message) {
    window.sessionStorage.test_failure = message;
  }
")]

extern "C" {
    #[wasm_bindgen()]
    fn test_success();

    #[wasm_bindgen()]
    fn test_failure(message: String);
}

#[wasm_bindgen]
pub async fn gpu_report() -> String {
    init_logger();

    let instance = wgpu::Instance::new({
        let mut desc = wgpu::InstanceDescriptor::default();
        desc.flags = wgpu::InstanceFlags::debugging();
        desc.backends = Backends::from_comma_list("gles");
        desc
    });

    let canvas = initialize_html_canvas();

    let surface = Some(
        instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .expect("could not create surface from canvas"),
    );

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: surface.as_ref(),
            ..Default::default()
        })
        .await
        .unwrap();

    log::info!("{:?}", adapter);

    let features = adapter.features();
    let limits = adapter.limits();
    let downlevel_caps = adapter.get_downlevel_capabilities();
    let texture_format_features = TEXTURE_FORMAT_LIST
        .into_iter()
        .map(|format| (format, adapter.get_texture_format_features(format)))
        .collect();

    let report = AdapterReport {
        info: adapter.get_info(),
        features,
        limits,
        downlevel_caps,
        texture_format_features,
    };

    let report = GpuReport {
        devices: vec![report],
    };

    serde_json::to_string_pretty(&report).expect("Failed to generate gpu report")
}

pub fn main(initializers: Vec<GpuTestInitializer>, test_name: String) {
    std::panic::set_hook(Box::new(|e| {
        test_failure(format!("{}", e));
    }));

    wasm_bindgen_futures::spawn_local(async move {
        for initializer in initializers {
            let test = initializer();
            if test.name == test_name {
                execute_test(None, test, None).await;
            }
        }

        test_success();
    });
}
