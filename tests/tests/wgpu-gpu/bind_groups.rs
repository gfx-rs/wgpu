use std::num::NonZeroU64;

use wgpu::{util::DeviceExt, BufferUsages, PollType};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        MULTIPLE_BINDINGS_WITH_DIFFERENT_SIZES,
        BIND_GROUP_NONFILTERING_LAYOUT_NONFILTERING_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MIN_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MAG_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MIPMAP_SAMPLER,
        BIND_GROUP_WITH_MAX_BINDING_INDEX,
        BIND_GROUP_STAGE_ORDER,
    ]);
}

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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
            uses_resource_table: false,
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

    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
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

#[apply(gpu_test!)]
static MULTIPLE_BINDINGS_WITH_DIFFERENT_SIZES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            .enable_noop(),
    )
    .run_sync(multiple_bindings_with_differing_sizes);

#[apply(gpu_test!)]
static BIND_GROUP_NONFILTERING_LAYOUT_NONFILTERING_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_nonfiltering_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                true,
            );
        });

#[apply(gpu_test!)]
static BIND_GROUP_NONFILTERING_LAYOUT_MIN_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_min_sampler"),
                    min_filter: wgpu::FilterMode::Linear,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[apply(gpu_test!)]
static BIND_GROUP_NONFILTERING_LAYOUT_MAG_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mag_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[apply(gpu_test!)]
static BIND_GROUP_NONFILTERING_LAYOUT_MIPMAP_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mipmap_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[apply(gpu_test!)]
static BIND_GROUP_WITH_MAX_BINDING_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits::downlevel_defaults()))
    .run_async(|ctx| async move {
        let (device, queue) = ctx
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_bindings_per_bind_group: ctx.adapter.limits().max_bindings_per_bind_group,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let max_binding_index = device.limits().max_bindings_per_bind_group - 1;
        let src_binding_index = max_binding_index - 1;
        let dst_binding_index = max_binding_index;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: src_binding_index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: dst_binding_index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
            uses_resource_table: false,
        });

        let shader = format!(
            "
            @group(0) @binding({src_binding_index}) var<uniform> src: u32;
            @group(0) @binding({dst_binding_index}) var<storage, read_write> dst: u32;
            @compute @workgroup_size(1)
            fn main() {{
                dst = src;
            }}"
        );

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            module: &module,
            cache: None,
        });

        let test_value = 123u32;

        let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::UNIFORM,
            contents: &test_value.to_le_bytes(),
        });
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: src_binding_index,
                    resource: wgpu::BindingResource::Buffer(src.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: dst_binding_index,
                    resource: wgpu::BindingResource::Buffer(dst.as_entire_buffer_binding()),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst, 0, &readback, 0, 4);
        queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        assert_eq!(
            &*readback.slice(..).get_mapped_range().unwrap(),
            &test_value.to_le_bytes()
        );
    });

/// Regression test for the wgpu-hal Metal backend binding a shader stage's
/// resources from the wrong offset into its internal per-bind-group
/// resource arrays.
///
/// wgpu-hal's Metal backend builds one flat array per resource kind
/// (buffers/textures/samplers) for each bind group, containing every
/// resource visible to any shader stage, laid out stage-by-stage in a fixed
/// order. When encoding `set_bind_group`, it computes where each stage's own
/// resources start in that array as the sum of the resource counts of every
/// stage that comes before it. If the stage order used to lay out the array
/// and the stage order used to sum up the preceding counts ever disagree,
/// a stage can end up bound to a completely different resource than the one
/// it was assigned in the bind group.
///
/// This creates a bind group with buffers visible to `TASK`, `MESH`, and
/// `FRAGMENT` (none of which the compute shader below actually uses) ahead
/// of a `COMPUTE`-visible buffer, then dispatches a compute shader that
/// doubles the compute buffer's value. If the compute stage's resources are
/// read from the wrong offset because of a stage-ordering mismatch, the
/// shader will double one of the filler buffers' values instead, and the
/// compute buffer will be left unmodified.
async fn bind_group_stage_order(ctx: TestingContext) {
    let device = &ctx.device;

    const FILLER_VALUE: u32 = 0xDEAD_0000;
    const COMPUTE_VALUE: u32 = 111;

    let make_filler_buffer = |label: &str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            usage: wgpu::BufferUsages::STORAGE,
            contents: bytemuck::bytes_of(&FILLER_VALUE),
        })
    };

    let task_filler = make_filler_buffer("task filler");
    let mesh_filler = make_filler_buffer("mesh filler");
    let fragment_filler = make_filler_buffer("fragment filler");
    let compute_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compute buffer"),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        contents: bytemuck::bytes_of(&COMPUTE_VALUE),
    });

    let filler_entry = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            filler_entry(0, wgpu::ShaderStages::TASK),
            filler_entry(1, wgpu::ShaderStages::MESH),
            filler_entry(2, wgpu::ShaderStages::FRAGMENT),
            filler_entry(3, wgpu::ShaderStages::COMPUTE),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: task_filler.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mesh_filler.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: fragment_filler.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: compute_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
        uses_resource_table: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(
            "
            @group(0) @binding(3) var<storage, read_write> value: u32;

            @compute @workgroup_size(1)
            fn main() {
                value = value * 2u;
            }
            "
            .into(),
        ),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&compute_buffer, 0, &readback, 0, 4);
    ctx.queue.submit(Some(encoder.finish()));

    readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    device.poll(PollType::wait_indefinitely()).unwrap();

    let result: u32 = *bytemuck::from_bytes(&readback.slice(..).get_mapped_range().unwrap());
    assert_eq!(
        result,
        COMPUTE_VALUE * 2,
        "compute stage was bound to the wrong buffer (expected the `COMPUTE`-visible buffer's \
         value to be doubled, got {result:#x})",
    );
}

#[apply(gpu_test!)]
static BIND_GROUP_STAGE_ORDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::EXPERIMENTAL_MESH_SHADER),
    )
    .run_async(bind_group_stage_order);
