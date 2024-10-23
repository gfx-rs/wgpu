use std::{
    num::{NonZeroU32, NonZeroU64},
    time::{Duration, Instant},
};

use criterion::{criterion_group, Criterion, Throughput};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::DeviceState;

fn dispatch_count() -> usize {
    // On CI we only want to run a very lightweight version of the benchmark
    // to ensure that it does not break.
    if std::env::var("WGPU_TESTING").is_ok() {
        8
    } else {
        10_000
    }
}

// Currently bindless is _much_ slower than with regularly resources,
// since wgpu needs to issues barriers for all resources between each dispatch for all read/write textures & buffers.
// This is in fact so slow that it makes the benchmark unusable when we use the same amount of
// resources as the regular benchmark.
// For details see https://github.com/gfx-rs/wgpu/issues/5766
fn dispatch_count_bindless() -> usize {
    // On CI we only want to run a very lightweight version of the benchmark
    // to ensure that it does not break.
    if std::env::var("WGPU_TESTING").is_ok() {
        8
    } else {
        1_000
    }
}

// Must match the number of textures in the computepass.wgsl shader
const TEXTURES_PER_DISPATCH: usize = 2;
const STORAGE_TEXTURES_PER_DISPATCH: usize = 2;
const STORAGE_BUFFERS_PER_DISPATCH: usize = 2;

const BUFFER_SIZE: u64 = 16;

struct ComputepassState {
    device_state: DeviceState,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,

    // Bindless resources
    bindless_bind_group: Option<wgpu::BindGroup>,
    bindless_pipeline: Option<wgpu::ComputePipeline>,
}

impl ComputepassState {
    /// Create and prepare all the resources needed for the computepass benchmark.
    fn new() -> Self {
        let device_state = DeviceState::new();

        let dispatch_count = dispatch_count();
        let dispatch_count_bindless = dispatch_count_bindless();
        let texture_count = dispatch_count * TEXTURES_PER_DISPATCH;
        let storage_buffer_count = dispatch_count * STORAGE_BUFFERS_PER_DISPATCH;
        let storage_texture_count = dispatch_count * STORAGE_TEXTURES_PER_DISPATCH;

        let supports_bindless = device_state.device.features().contains(
            wgpu::Features::BUFFER_BINDING_ARRAY
                | wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
        // TODO: as of writing llvmpipe segfaults the bindless benchmark on ci
        && device_state.adapter_info.driver != "llvmpipe";

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let mut bind_group_layout_entries = Vec::with_capacity(TEXTURES_PER_DISPATCH);
        for i in 0..TEXTURES_PER_DISPATCH {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for i in 0..STORAGE_TEXTURES_PER_DISPATCH {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (TEXTURES_PER_DISPATCH + i) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
        }
        for i in 0..STORAGE_BUFFERS_PER_DISPATCH {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (TEXTURES_PER_DISPATCH + STORAGE_BUFFERS_PER_DISPATCH + i) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(BUFFER_SIZE),
                },
                count: None,
            });
        }

        let bind_group_layout =
            device_state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries,
                });

        let mut texture_views = Vec::with_capacity(texture_count);
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
        let texture_view_refs: Vec<_> = texture_views.iter().collect();

        let mut storage_texture_views = Vec::with_capacity(storage_texture_count);
        for i in 0..storage_texture_count {
            let texture = device_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("StorageTexture {i}")),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
            storage_texture_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("StorageTexture View {i}")),
                ..Default::default()
            }));
        }
        random.shuffle(&mut storage_texture_views);
        let storage_texture_view_refs: Vec<_> = storage_texture_views.iter().collect();

        let mut storage_buffers = Vec::with_capacity(storage_buffer_count);
        for i in 0..storage_buffer_count {
            storage_buffers.push(device_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Buffer {i}")),
                size: BUFFER_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        random.shuffle(&mut storage_buffers);
        let storage_buffer_bindings: Vec<_> = storage_buffers
            .iter()
            .map(|b| b.as_entire_buffer_binding())
            .collect();

        let mut bind_groups = Vec::with_capacity(dispatch_count);
        for dispatch_idx in 0..dispatch_count {
            let mut entries = Vec::with_capacity(TEXTURES_PER_DISPATCH);
            for tex_idx in 0..TEXTURES_PER_DISPATCH {
                entries.push(wgpu::BindGroupEntry {
                    binding: tex_idx as u32,
                    resource: wgpu::BindingResource::TextureView(
                        &texture_views[dispatch_idx * TEXTURES_PER_DISPATCH + tex_idx],
                    ),
                });
            }
            for tex_idx in 0..STORAGE_TEXTURES_PER_DISPATCH {
                entries.push(wgpu::BindGroupEntry {
                    binding: (TEXTURES_PER_DISPATCH + tex_idx) as u32,
                    resource: wgpu::BindingResource::TextureView(
                        &storage_texture_views
                            [dispatch_idx * STORAGE_TEXTURES_PER_DISPATCH + tex_idx],
                    ),
                });
            }
            for buffer_idx in 0..STORAGE_BUFFERS_PER_DISPATCH {
                entries.push(wgpu::BindGroupEntry {
                    binding: (TEXTURES_PER_DISPATCH + STORAGE_BUFFERS_PER_DISPATCH + buffer_idx)
                        as u32,
                    resource: wgpu::BindingResource::Buffer(
                        storage_buffers[dispatch_idx * STORAGE_BUFFERS_PER_DISPATCH + buffer_idx]
                            .as_entire_buffer_binding(),
                    ),
                });
            }

            bind_groups.push(
                device_state
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bind_group_layout,
                        entries: &entries,
                    }),
            );
        }
        random.shuffle(&mut bind_groups);

        let sm = device_state
            .device
            .create_shader_module(wgpu::include_wgsl!("computepass.wgsl"));

        let pipeline_layout =
            device_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            device_state
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &sm,
                    entry_point: Some("cs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let (bindless_bind_group, bindless_pipeline) = if supports_bindless {
            let bindless_bind_group_layout =
                device_state
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: Some(NonZeroU32::new(texture_count as u32).unwrap()),
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::ReadWrite,
                                    format: wgpu::TextureFormat::R32Float,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: Some(NonZeroU32::new(storage_texture_count as u32).unwrap()),
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: std::num::NonZeroU64::new(BUFFER_SIZE),
                                },
                                count: Some(NonZeroU32::new(storage_buffer_count as u32).unwrap()),
                            },
                        ],
                    });

            let bindless_bind_group =
                device_state
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bindless_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureViewArray(
                                    &texture_view_refs[..dispatch_count_bindless],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureViewArray(
                                    &storage_texture_view_refs[..dispatch_count_bindless],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::BufferArray(
                                    &storage_buffer_bindings[..dispatch_count_bindless],
                                ),
                            },
                        ],
                    });

            let bindless_sm = device_state
                .device
                .create_shader_module(wgpu::include_wgsl!("computepass-bindless.wgsl"));

            let bindless_pipeline_layout =
                device_state
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bindless_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let bindless_pipeline =
                device_state
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Compute Pipeline bindless"),
                        layout: Some(&bindless_pipeline_layout),
                        module: &bindless_sm,
                        entry_point: Some("cs_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            (Some(bindless_bind_group), Some(bindless_pipeline))
        } else {
            (None, None)
        };

        Self {
            device_state,
            pipeline,
            bind_groups,

            bindless_bind_group,
            bindless_pipeline,
        }
    }

    fn run_subpass(&self, pass_number: usize, total_passes: usize) -> wgpu::CommandBuffer {
        profiling::scope!("Computepass", &format!("Pass {pass_number}/{total_passes}"));

        let dispatch_count = dispatch_count();
        let dispatch_per_pass = dispatch_count / total_passes;

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        let start_idx = pass_number * dispatch_per_pass;
        let end_idx = start_idx + dispatch_per_pass;
        for dispatch_idx in start_idx..end_idx {
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups[dispatch_idx], &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        drop(compute_pass);

        encoder.finish()
    }

    fn run_bindless_pass(&self, dispatch_count_bindless: usize) -> wgpu::CommandBuffer {
        profiling::scope!("Bindless Computepass");

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(self.bindless_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, Some(self.bindless_bind_group.as_ref().unwrap()), &[]);
        for _ in 0..dispatch_count_bindless {
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        drop(compute_pass);

        encoder.finish()
    }
}

fn run_bench(ctx: &mut Criterion) {
    let state = Lazy::new(ComputepassState::new);

    let dispatch_count = dispatch_count();
    let dispatch_count_bindless = dispatch_count_bindless();
    let texture_count = dispatch_count * TEXTURES_PER_DISPATCH;
    let storage_buffer_count = dispatch_count * STORAGE_BUFFERS_PER_DISPATCH;
    let storage_texture_count = dispatch_count * STORAGE_TEXTURES_PER_DISPATCH;

    // Test 10k dispatch calls split up into 1, 2, 4, and 8 computepasses
    let mut group = ctx.benchmark_group("Computepass: Single Threaded");
    group.throughput(Throughput::Elements(dispatch_count as _));

    for time_submit in [false, true] {
        for cpasses in [1, 2, 4, 8] {
            let dispatch_per_pass = dispatch_count / cpasses;

            let label = if time_submit {
                "Submit Time"
            } else {
                "Computepass Time"
            };

            group.bench_function(
                format!("{cpasses} computepasses x {dispatch_per_pass} dispatches ({label})"),
                |b| {
                    Lazy::force(&state);

                    b.iter_custom(|iters| {
                        profiling::scope!("benchmark invocation");

                        let mut duration = Duration::ZERO;

                        for _ in 0..iters {
                            profiling::scope!("benchmark iteration");

                            let mut start = Instant::now();

                            let mut buffers: Vec<wgpu::CommandBuffer> = Vec::with_capacity(cpasses);
                            for i in 0..cpasses {
                                buffers.push(state.run_subpass(i, cpasses));
                            }

                            if time_submit {
                                start = Instant::now();
                            } else {
                                duration += start.elapsed();
                            }

                            state.device_state.queue.submit(buffers);

                            if time_submit {
                                duration += start.elapsed();
                            }

                            state.device_state.device.poll(wgpu::Maintain::Wait);
                        }

                        duration
                    })
                },
            );
        }
    }
    group.finish();

    // Test 10k dispatch calls split up over 2, 4, and 8 threads.
    let mut group = ctx.benchmark_group("Computepass: Multi Threaded");
    group.throughput(Throughput::Elements(dispatch_count as _));

    for threads in [2, 4, 8] {
        let dispatch_per_pass = dispatch_count / threads;
        group.bench_function(
            format!("{threads} threads x {dispatch_per_pass} dispatch"),
            |b| {
                Lazy::force(&state);

                b.iter_custom(|iters| {
                    profiling::scope!("benchmark invocation");

                    // This benchmark hangs on Apple Paravirtualized GPUs. No idea why.
                    if state.device_state.adapter_info.name.contains("Paravirtual") {
                        return Duration::from_secs_f32(1.0);
                    }

                    let mut duration = Duration::ZERO;

                    for _ in 0..iters {
                        profiling::scope!("benchmark iteration");

                        let start = Instant::now();

                        let buffers = (0..threads)
                            .into_par_iter()
                            .map(|i| state.run_subpass(i, threads))
                            .collect::<Vec<_>>();

                        duration += start.elapsed();

                        state.device_state.queue.submit(buffers);
                        state.device_state.device.poll(wgpu::Maintain::Wait);
                    }

                    duration
                })
            },
        );
    }
    group.finish();

    // Test 10k dispatch calls split up over 1, 2, 4, and 8 threads.
    let mut group = ctx.benchmark_group("Computepass: Bindless");
    group.throughput(Throughput::Elements(dispatch_count_bindless as _));

    group.bench_function(format!("{dispatch_count_bindless} dispatch"), |b| {
        Lazy::force(&state);

        b.iter_custom(|iters| {
            profiling::scope!("benchmark invocation");

            // This benchmark hangs on Apple Paravirtualized GPUs. No idea why.
            if state.device_state.adapter_info.name.contains("Paravirtual") {
                return Duration::from_secs_f32(1.0);
            }

            // Need bindless to run this benchmark
            if state.bindless_bind_group.is_none() {
                return Duration::from_secs_f32(1.0);
            }

            let mut duration = Duration::ZERO;

            for _ in 0..iters {
                profiling::scope!("benchmark iteration");

                let start = Instant::now();

                let buffer = state.run_bindless_pass(dispatch_count_bindless);

                duration += start.elapsed();

                state.device_state.queue.submit([buffer]);
                state.device_state.device.poll(wgpu::Maintain::Wait);
            }

            duration
        })
    });
    group.finish();

    ctx.bench_function(
        &format!(
            "Computepass: Empty Submit with {} Resources",
            texture_count + storage_texture_count + storage_buffer_count
        ),
        |b| {
            Lazy::force(&state);

            b.iter(|| state.device_state.queue.submit([]));
        },
    );
}

criterion_group! {
    name = computepass;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = run_bench,
}
