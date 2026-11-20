use std::time::Instant;

use wgpu_benchmark::{iter, BenchmarkContext, SubBenchResult};

use crate::DeviceState;

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let state = DeviceState::new();
    let mut results = Vec::new();
    const COPIES: u32 = 1_000;

    for layers in [1, 8] {
        let size = wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: layers,
        };
        let desc = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let src = state.device.create_texture(&desc);
        let dst = state.device.create_texture(&desc);

        results.push(iter(
            &ctx,
            &format!("{layers} regions"),
            "copies",
            COPIES,
            || {
                let mut encoder = state.device.create_command_encoder(&Default::default());
                let start = Instant::now();
                for _ in 0..COPIES {
                    encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), size);
                }
                let commands = encoder.finish();
                let duration = start.elapsed();
                drop(commands);
                duration
            },
        ));
    }

    Ok(results)
}
