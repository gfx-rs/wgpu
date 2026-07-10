/// Tests for `max_buffers_and_acceleration_structures_per_shader_stage`.
///
/// This limit only has a meaningful finite value on Metal (non-strict mode), where all buffer
/// types (storage, uniform, vertex buffers, acceleration structures) share a single argument
/// table of 31 slots with 2 reserved for internal use, leaving 29 user-visible slots.
///
/// On all other backends the limit is `u32::MAX`, so these tests are skipped there to avoid
/// hitting other, lower per-type limits while trying to reach the combined limit.
use wgpu_test::{
    fail, gpu_test, valid, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        BUFFERS_AND_ACCEL_STRUCTS_VALID_WITHIN_LIMIT,
        BUFFERS_AND_ACCEL_STRUCTS_VALID_VERTEX_STAGE_AT_LIMIT,
        BUFFERS_AND_ACCEL_STRUCTS_EXCEEDS_SINGLE_BGL,
        BUFFERS_AND_ACCEL_STRUCTS_EXCEEDS_ACROSS_BGLS,
        BUFFERS_AND_ACCEL_STRUCTS_VERTEX_STAGE_EXCEEDS_WITH_VERTEX_BUFFERS,
    ]);
}

/// Skip on every backend except Metal. The combined limit is `u32::MAX` on other backends,
/// so reaching it would require an impractical number of resources and would run into the
/// individual per-type limits instead.
fn skip_non_metal() -> FailureCase {
    FailureCase::backend(!wgpu::Backends::METAL)
}

/// Device limits that reflect Metal's combined buffer slot constraint.
///
/// `max_buffers_and_acceleration_structures_per_shader_stage` is set to 29 (Metal's limit).
/// The per-type limits are raised above the default 8/12 so that tests can exercise the
/// *combined* limit without being blocked by an individual per-type limit first.
///
/// Setting `max_storage_buffers_per_shader_stage: 29` also means the test is automatically
/// skipped on Metal in STRICT_WEBGPU_COMPLIANCE mode, where the adapter only supports 8.
fn metal_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 29,
        max_uniform_buffers_per_shader_stage: 29,
        max_vertex_buffers: 16,
        max_buffers_and_acceleration_structures_per_shader_stage: 29,
        ..wgpu::Limits::defaults()
    }
}

fn storage_entries(n: u32, visibility: wgpu::ShaderStages) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..n)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn uniform_entries(
    n: u32,
    binding_offset: u32,
    visibility: wgpu::ShaderStages,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..n)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: binding_offset + i,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

/// 14 storage + 14 uniform = 28 combined in a BGL.
/// Pipeline layout creation must succeed since 28 < 29.
#[gpu_test]
static BUFFERS_AND_ACCEL_STRUCTS_VALID_WITHIN_LIMIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(metal_limits())
                .skip(skip_non_metal()),
        )
        .run_sync(|ctx| {
            let mut entries = storage_entries(14, wgpu::ShaderStages::COMPUTE);
            entries.extend(uniform_entries(14, 14, wgpu::ShaderStages::COMPUTE));

            valid(&ctx.device, || {
                let bgl = ctx
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &entries,
                    });
                let _ = ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    });
            });
        });

/// 20 storage buffers + 9 vertex buffers = 29 combined.
/// Render pipeline creation must succeed since 29 == limit.
#[gpu_test]
static BUFFERS_AND_ACCEL_STRUCTS_VALID_VERTEX_STAGE_AT_LIMIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(metal_limits())
                .skip(skip_non_metal()),
        )
        .run_sync(|ctx| {
            let entries = storage_entries(20, wgpu::ShaderStages::VERTEX);

            let bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    });

            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
                         @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }"
                            .into(),
                    ),
                });

            const EMPTY_VB: Option<wgpu::VertexBufferLayout<'static>> =
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[],
                });

            valid(&ctx.device, || {
                let _ = ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs"),
                            compilation_options: Default::default(),
                            buffers: &[EMPTY_VB; 9],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    });
            });
        });

/// 15 storage + 15 uniform = 30 combined in a single BGL.
/// Must fail at BGL creation since the combined limit is 29.
#[gpu_test]
static BUFFERS_AND_ACCEL_STRUCTS_EXCEEDS_SINGLE_BGL: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(metal_limits())
                .skip(skip_non_metal()),
        )
        .run_sync(|ctx| {
            let mut entries = storage_entries(15, wgpu::ShaderStages::COMPUTE);
            entries.extend(uniform_entries(15, 15, wgpu::ShaderStages::COMPUTE));

            fail(
                &ctx.device,
                || {
                    let _ = ctx
                        .device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: None,
                            entries: &entries,
                        });
                },
                Some("max_buffers_and_acceleration_structures_per_shader_stage"),
            );
        });

/// Two BGLs where one has 15 storage buffers and the other has 15 uniform
/// buffers. Each individual BGL is within the combined limit, and neither
/// per-type limit is exceeded across both BGLs. However their combined use
/// in a pipeline layout adds up to 30 total buffer slots, exceeding the limit.
#[gpu_test]
static BUFFERS_AND_ACCEL_STRUCTS_EXCEEDS_ACROSS_BGLS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(metal_limits())
                .skip(skip_non_metal()),
        )
        .run_sync(|ctx| {
            let storage = storage_entries(15, wgpu::ShaderStages::COMPUTE);
            let uniform = uniform_entries(15, 0, wgpu::ShaderStages::COMPUTE);

            let bgl_a = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bgl_a"),
                    entries: &storage,
                });
            let bgl_b = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bgl_b"),
                    entries: &uniform,
                });

            fail(
                &ctx.device,
                || {
                    let _ = ctx
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[Some(&bgl_a), Some(&bgl_b)],
                            immediate_size: 0,
                        });
                },
                Some("max_buffers_and_acceleration_structures_per_shader_stage"),
            );
        });

/// 20 storage buffers + 10 vertex buffers = 30 combined in the vertex stage,
/// which exceeds the limit of 29. Render pipeline creation must fail.
#[gpu_test]
static BUFFERS_AND_ACCEL_STRUCTS_VERTEX_STAGE_EXCEEDS_WITH_VERTEX_BUFFERS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(metal_limits())
                .skip(skip_non_metal()),
        )
        .run_sync(|ctx| {
            let entries = storage_entries(20, wgpu::ShaderStages::VERTEX);

            let bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    });

            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
                         @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }"
                            .into(),
                    ),
                });

            const EMPTY_VB: Option<wgpu::VertexBufferLayout<'static>> =
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[],
                });

            fail(
                &ctx.device,
                || {
                    let _ = ctx
                        .device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: None,
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs"),
                                compilation_options: Default::default(),
                                buffers: &[EMPTY_VB; 10],
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs"),
                                compilation_options: Default::default(),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: wgpu::TextureFormat::Rgba8Unorm,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                            }),
                            primitive: wgpu::PrimitiveState::default(),
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState::default(),
                            multiview_mask: None,
                            cache: None,
                        });
                },
                Some("vertex-stage buffers and acceleration structures"),
            );
        });
