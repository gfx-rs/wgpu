use std::num::NonZeroU64;

use wgpu::{BufferUsages, PollType};
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

/// Create two bind groups against the same bind group layout, in the same
/// compute pass, but against two different shaders that have different binding
/// sizes. The first has binding size 8, the second has binding size 4.
///
/// Regression test for https://github.com/gfx-rs/wgpu/issues/7359.
fn multiple_bindings_with_differing_sizes(ctx: TestingContext) {
    const SHADER_SRC: &[&str] = &[
        "
        @group(0) @binding(0)
        var<uniform> buffer : vec2<f32>;

        @compute @workgroup_size(1, 1, 1) fn main() {
            // Just need a static use.
            let _value = buffer.x;
        }
        ",
        "
        @group(0) @binding(0)
        var<uniform> buffer : f32;

        @compute @workgroup_size(1, 1, 1) fn main() {
            // Just need a static use.
            let _value = buffer;
        }
        ",
    ];

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer"),
        size: 8,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipelines = SHADER_SRC
        .iter()
        .enumerate()
        .map(|(i, &shader_src)| {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("shader{i}")),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("pipeline{i}")),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
        .collect::<Vec<_>>();

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

    for (i, pipeline) in pipelines.iter().enumerate() {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bg{i}")),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(u64::try_from(8 - 4 * i).unwrap()).unwrap()),
                }),
            }],
        });

        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[0]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    drop(cpass);

    let data = [0u8; 8];
    ctx.queue.write_buffer(&buffer, 0, &data);
    ctx.queue.submit(Some(encoder.finish()));

    ctx.device.poll(PollType::Wait).unwrap();
}

/// Test `descriptor` against a bind group layout that requires non-filtering sampler.
fn try_sampler_nonfiltering_layout(
    ctx: TestingContext,
    descriptor: &wgpu::SamplerDescriptor,
    good: bool,
) {
    let label = descriptor.label;
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            }],
        });

    let sampler = ctx.device.create_sampler(descriptor);

    let create_bind_group = || {
        let _ = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            }],
        });
    };

    if good {
        wgpu_test::valid(&ctx.device, create_bind_group);
    } else {
        wgpu_test::fail(
            &ctx.device,
            create_bind_group,
            Some("but given a sampler with filtering"),
        );
    }
}

#[gpu_test]
static MULTIPLE_BINDINGS_WITH_DIFFERENT_SIZES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            .expect_fail(FailureCase::always()), // https://github.com/gfx-rs/wgpu/issues/7359
    )
    .run_sync(multiple_bindings_with_differing_sizes);

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_NONFILTERING_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_nonfiltering_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                true,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MIN_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_min_sampler"),
                    min_filter: wgpu::FilterMode::Linear,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MAG_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mag_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MIPMAP_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mipmap_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });
