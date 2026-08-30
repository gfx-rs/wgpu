//! Feature / limit coverage:
//!
//! * a table created at the maximum size, with a texture bound into a high slot
//!   and sampled,
//! * ordinary bind groups (groups 0 and 1) coexisting with a resource table
//!   (bound at set index 2, = the layout's bind-group count), and
//! * `getResource` invoked from a WGSL helper function rather than directly in
//!   the entry point.

use wgpu::*;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer};

use super::common::{make_red_texture, read_u32s, run_sampling, table_params, texture_red};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_NEAR_MAX_SIZE,
        RESOURCE_TABLE_WITH_BIND_GROUPS,
        RESOURCE_TABLE_HELPER_FUNCTION,
    ]);
}

/// The maximum resource-table size (matches `wgpu_core`'s
/// `MAX_RESOURCE_TABLE_SIZE`; the Vulkan update-after-bind sampled-image limits
/// are far higher, so every supported adapter clamps here rather than lower).
const MAX_TABLE_SIZE: u32 = 65536;

/// Create a maximum-size table, bind a texture into the highest slot, and sample
/// it. Exercises both the large allocation and a high (variable-count) array
/// index.
#[apply(gpu_test!)]
static RESOURCE_TABLE_NEAR_MAX_SIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(0);
        const HIGH_SLOT: u32 = MAX_TABLE_SIZE - 1;

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: Some("max-size table"),
            size: MAX_TABLE_SIZE,
        });
        assert_eq!(table.size(), MAX_TABLE_SIZE);

        table.update(HIGH_SLOT, &view).expect("bind into high slot");

        let got = run_sampling(&ctx, &table, &[HIGH_SLOT]).await;
        assert_eq!(got, vec![RED as u32]);
    });

/// A pipeline that uses ordinary bind groups (group 0 for the indices/output
/// storage buffers, group 1 for a uniform bias) *and* a resource table. With two
/// bind-group layouts the table binds at set index 2. The output combines the
/// sampled texel with the group-1 bias, so a correct result proves all three
/// descriptor sets were bound and read.
#[apply(gpu_test!)]
static RESOURCE_TABLE_WITH_BIND_GROUPS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(2); // 30
        const BIAS: u32 = 100;

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("bind");

        let shader = r#"
enable resource_table;
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(1) @binding(0) var<uniform> bias: u32;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let tex = getResource<texture_2d<f32>>(indices[i]);
    let texel = textureLoad(tex, vec2<i32>(0, 0), 0);
    output[i] = u32(round(texel.r * 255.0)) + bias;
}
"#;
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader.into()),
        });

        let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&index_buffer, 0, &0u32.to_ne_bytes());
        let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bias_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&bias_buffer, 0, &BIAS.to_ne_bytes());

        let bgl0 = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bgl1 = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bg0 = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl0,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let bg1 = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl1,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: bias_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
                immediate_size: 0,
                uses_resource_table: true,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_resource_table(Some(&table));
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let got = read_u32s(&ctx, &readback_buffer).await;
        assert_eq!(
            got,
            vec![RED as u32 + BIAS],
            "output should combine the sampled texel with the group-1 bias"
        );
    });

/// `getResource` invoked from a helper function reached from the entry point,
/// verified on the GPU. Exercises the SPIR-V backend's over-listing of the
/// synthesized table globals into the entry-point interface (they are read only
/// inside the called helper).
#[apply(gpu_test!)]
static RESOURCE_TABLE_HELPER_FUNCTION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(3); // 40

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("bind");

        let shader = r#"
enable resource_table;
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn sample_red(slot: u32) -> f32 {
    let tex = getResource<texture_2d<f32>>(slot);
    return textureLoad(tex, vec2<i32>(0, 0), 0).r;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    output[i] = u32(round(sample_red(indices[i]) * 255.0));
}
"#;
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader.into()),
        });

        let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&index_buffer, 0, &0u32.to_ne_bytes());
        let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
                uses_resource_table: true,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_resource_table(Some(&table));
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let got = read_u32s(&ctx, &readback_buffer).await;
        assert_eq!(got, vec![RED as u32]);
    });
