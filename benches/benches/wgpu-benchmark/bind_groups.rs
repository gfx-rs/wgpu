use std::{
    num::NonZeroU32,
    time::{Duration, Instant},
};

use criterion::{criterion_group, Criterion, Throughput};
use nanorand::{Rng, WyRand};
use std::sync::LazyLock;

use crate::{is_test, DeviceState};

// Creating 50_000 textures takes a considerable amount of time with syncval enabled.
//
// We greatly reduce the number of textures for the test case to keep the runtime
// reasonable for testing.
const MAX_TEXTURE_COUNT_BENCHMARK: u32 = 50_000;
const TEXTURE_COUNTS_BENCHMARK: &[u32] = &[5, 50, 500, 5_000, 50_000];

const MAX_TEXTURE_COUNT_TEST: u32 = 5;
const TEXTURE_COUNTS_TEST: &[u32] = &[5];

struct BindGroupState {
    device_state: DeviceState,
    texture_views: Vec<wgpu::TextureView>,
}

impl BindGroupState {
    /// Create and prepare all the resources needed for the renderpass benchmark.
    fn new() -> Self {
        let device_state = DeviceState::new();

        let texture_count = if is_test() {
            MAX_TEXTURE_COUNT_TEST
        } else {
            MAX_TEXTURE_COUNT_BENCHMARK
        };

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let mut texture_views = Vec::with_capacity(texture_count as usize);
        for i in 0..texture_count {
            let texture = device_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Texture {i}")),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            texture_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Texture View {i}")),
                ..Default::default()
            }));
        }
        random.shuffle(&mut texture_views);

        Self {
            device_state,
            texture_views,
        }
    }
}

fn run_bench(ctx: &mut Criterion) {
    let state = LazyLock::new(BindGroupState::new);

    let mut group = ctx.benchmark_group("Bind Group Creation");

    let count_list = if is_test() {
        TEXTURE_COUNTS_TEST
    } else {
        TEXTURE_COUNTS_BENCHMARK
    };

    for &count in count_list {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            format!("{count} Element Bind Group"),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    if !state
                        .device_state
                        .device
                        .features()
                        .contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
                    {
                        return Duration::ZERO;
                    }

                    if count
                        > state
                            .device_state
                            .device
                            .limits()
                            .max_sampled_textures_per_shader_stage
                    {
                        return Duration::ZERO;
                    }

                    let bind_group_layout = state.device_state.device.create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: None,
                            entries: &[wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: Some(NonZeroU32::new(count).unwrap()),
                            }],
                        },
                    );

                    let texture_view_refs: Vec<_> =
                        state.texture_views.iter().take(count as usize).collect();

                    let mut duration = Duration::ZERO;
                    for _ in 0..iters {
                        profiling::scope!("benchmark iteration");

                        let start = Instant::now();
                        let bind_group = state.device_state.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                layout: &bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureViewArray(
                                        &texture_view_refs,
                                    ),
                                }],
                                label: None,
                            },
                        );

                        duration += start.elapsed();

                        drop(bind_group);
                        state
                            .device_state
                            .device
                            .poll(wgpu::PollType::Wait)
                            .unwrap();
                    }

                    duration
                });
            },
        );
    }
}

criterion_group! {
    name = bind_groups;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = run_bench,
}
