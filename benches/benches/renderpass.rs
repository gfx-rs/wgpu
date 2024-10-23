use std::{
    num::NonZeroU32,
    time::{Duration, Instant},
};

use criterion::{criterion_group, Criterion, Throughput};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::DeviceState;

fn draw_count() -> usize {
    // On CI we only want to run a very lightweight version of the benchmark
    // to ensure that it does not break.
    if std::env::var("WGPU_TESTING").is_ok() {
        8
    } else {
        10_000
    }
}

// Must match the number of textures in the renderpass.wgsl shader
const TEXTURES_PER_DRAW: usize = 7;
const VERTEX_BUFFERS_PER_DRAW: usize = 2;

struct RenderpassState {
    device_state: DeviceState,
    pipeline: wgpu::RenderPipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    render_target: wgpu::TextureView,

    // Bindless resources
    bindless_bind_group: Option<wgpu::BindGroup>,
    bindless_pipeline: Option<wgpu::RenderPipeline>,
}

impl RenderpassState {
    /// Create and prepare all the resources needed for the renderpass benchmark.
    fn new() -> Self {
        let device_state = DeviceState::new();

        let draw_count = draw_count();
        let vertex_buffer_count = draw_count * VERTEX_BUFFERS_PER_DRAW;
        let texture_count = draw_count * TEXTURES_PER_DRAW;

        let supports_bindless = device_state.device.features().contains(
            wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        ) && device_state
            .device
            .limits()
            .max_sampled_textures_per_shader_stage
            >= texture_count as _;

        // Performance gets considerably worse if the resources are shuffled.
        //
        // This more closely matches the real-world use case where resources have no
        // well defined usage order.
        let mut random = WyRand::new_seed(0x8BADF00D);

        let mut bind_group_layout_entries = Vec::with_capacity(TEXTURES_PER_DRAW);
        for i in 0..TEXTURES_PER_DRAW {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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

        let mut bind_groups = Vec::with_capacity(draw_count);
        for draw_idx in 0..draw_count {
            let mut entries = Vec::with_capacity(TEXTURES_PER_DRAW);
            for tex_idx in 0..TEXTURES_PER_DRAW {
                entries.push(wgpu::BindGroupEntry {
                    binding: tex_idx as u32,
                    resource: wgpu::BindingResource::TextureView(
                        &texture_views[draw_idx * TEXTURES_PER_DRAW + tex_idx],
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
            .create_shader_module(wgpu::include_wgsl!("renderpass.wgsl"));

        let pipeline_layout =
            device_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let mut vertex_buffers = Vec::with_capacity(vertex_buffer_count);
        for _ in 0..vertex_buffer_count {
            vertex_buffers.push(device_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 3 * 16,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            }));
        }
        random.shuffle(&mut vertex_buffers);

        let mut index_buffers = Vec::with_capacity(draw_count);
        for _ in 0..draw_count {
            index_buffers.push(device_state.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 3 * 4,
                usage: wgpu::BufferUsages::INDEX,
                mapped_at_creation: false,
            }));
        }
        random.shuffle(&mut index_buffers);

        let mut vertex_buffer_attributes = Vec::with_capacity(VERTEX_BUFFERS_PER_DRAW);
        for i in 0..VERTEX_BUFFERS_PER_DRAW {
            vertex_buffer_attributes.push(wgpu::vertex_attr_array![i as u32 => Float32x4]);
        }

        let mut vertex_buffer_layouts = Vec::with_capacity(VERTEX_BUFFERS_PER_DRAW);
        for attributes in &vertex_buffer_attributes {
            vertex_buffer_layouts.push(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes,
            });
        }

        let pipeline =
            device_state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &sm,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffer_layouts,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Cw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &sm,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    multiview: None,
                    cache: None,
                });

        let render_target = device_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Render Target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut bindless_bind_group = None;
        let mut bindless_pipeline = None;

        if supports_bindless {
            let bindless_bind_group_layout =
                device_state
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: Some(NonZeroU32::new(texture_count as u32).unwrap()),
                        }],
                    });

            bindless_bind_group = Some(device_state.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bindless_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_view_refs),
                    }],
                },
            ));

            let bindless_shader_module = device_state
                .device
                .create_shader_module(wgpu::include_wgsl!("renderpass-bindless.wgsl"));

            let bindless_pipeline_layout =
                device_state
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bindless_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            bindless_pipeline = Some(device_state.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&bindless_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &bindless_shader_module,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffer_layouts,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Cw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &bindless_shader_module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    multiview: None,
                    cache: None,
                },
            ));
        }

        Self {
            device_state,
            pipeline,
            bind_groups,
            vertex_buffers,
            index_buffers,
            render_target,

            bindless_bind_group,
            bindless_pipeline,
        }
    }

    fn run_subpass(
        &self,
        pass_number: usize,
        total_passes: usize,
        draw_count: usize,
    ) -> wgpu::CommandBuffer {
        profiling::scope!("Renderpass", &format!("Pass {pass_number}/{total_passes}"));

        let draws_per_pass = draw_count / total_passes;

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            occlusion_query_set: None,
            timestamp_writes: None,
            depth_stencil_attachment: None,
        });

        let start_idx = pass_number * draws_per_pass;
        let end_idx = start_idx + draws_per_pass;
        for draw_idx in start_idx..end_idx {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_groups[draw_idx], &[]);
            for i in 0..VERTEX_BUFFERS_PER_DRAW {
                render_pass.set_vertex_buffer(
                    i as u32,
                    self.vertex_buffers[draw_idx * VERTEX_BUFFERS_PER_DRAW + i].slice(..),
                );
            }
            render_pass.set_index_buffer(
                self.index_buffers[draw_idx].slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..3, 0, 0..1);
        }

        drop(render_pass);

        encoder.finish()
    }

    fn run_bindless_pass(&self, draw_count: usize) -> wgpu::CommandBuffer {
        profiling::scope!("Bindless Renderpass");

        let mut encoder = self
            .device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            occlusion_query_set: None,
            timestamp_writes: None,
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(self.bindless_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, Some(self.bindless_bind_group.as_ref().unwrap()), &[]);
        for i in 0..VERTEX_BUFFERS_PER_DRAW {
            render_pass.set_vertex_buffer(i as u32, self.vertex_buffers[0].slice(..));
        }
        render_pass.set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint32);

        for draw_idx in 0..draw_count {
            render_pass.draw_indexed(0..3, 0, draw_idx as u32..draw_idx as u32 + 1);
        }

        drop(render_pass);

        encoder.finish()
    }
}

fn run_bench(ctx: &mut Criterion) {
    let state = Lazy::new(RenderpassState::new);

    let draw_count = draw_count();
    let vertex_buffer_count = draw_count * VERTEX_BUFFERS_PER_DRAW;
    let texture_count = draw_count * TEXTURES_PER_DRAW;

    // Test 10k draw calls split up into 1, 2, 4, and 8 renderpasses
    let mut group = ctx.benchmark_group("Renderpass: Single Threaded");
    group.throughput(Throughput::Elements(draw_count as _));

    for time_submit in [false, true] {
        for rpasses in [1, 2, 4, 8] {
            let draws_per_pass = draw_count / rpasses;

            let label = if time_submit {
                "Submit Time"
            } else {
                "Renderpass Time"
            };

            group.bench_function(
                format!("{rpasses} renderpasses x {draws_per_pass} draws ({label})"),
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

                            let mut start = Instant::now();

                            let mut buffers: Vec<wgpu::CommandBuffer> = Vec::with_capacity(rpasses);
                            for i in 0..rpasses {
                                buffers.push(state.run_subpass(i, rpasses, draw_count));
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

    // Test 10k draw calls split up over 2, 4, and 8 threads.
    let mut group = ctx.benchmark_group("Renderpass: Multi Threaded");
    group.throughput(Throughput::Elements(draw_count as _));

    for threads in [2, 4, 8] {
        let draws_per_pass = draw_count / threads;
        group.bench_function(format!("{threads} threads x {draws_per_pass} draws"), |b| {
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
                        .map(|i| state.run_subpass(i, threads, draw_count))
                        .collect::<Vec<_>>();

                    duration += start.elapsed();

                    state.device_state.queue.submit(buffers);
                    state.device_state.device.poll(wgpu::Maintain::Wait);
                }

                duration
            })
        });
    }
    group.finish();

    // Test 10k draw calls split up over 1, 2, 4, and 8 threads.
    let mut group = ctx.benchmark_group("Renderpass: Bindless");
    group.throughput(Throughput::Elements(draw_count as _));

    group.bench_function(format!("{draw_count} draws"), |b| {
        Lazy::force(&state);

        b.iter_custom(|iters| {
            profiling::scope!("benchmark invocation");

            // Need bindless to run this benchmark
            if state.bindless_bind_group.is_none() {
                return Duration::from_secs_f32(1.0);
            }

            let mut duration = Duration::ZERO;

            for _ in 0..iters {
                profiling::scope!("benchmark iteration");

                let start = Instant::now();

                let buffer = state.run_bindless_pass(draw_count);

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
            "Renderpass: Empty Submit with {} Resources",
            texture_count + vertex_buffer_count
        ),
        |b| {
            Lazy::force(&state);

            b.iter(|| state.device_state.queue.submit([]));
        },
    );
}

criterion_group! {
    name = renderpass;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = run_bench,
}
