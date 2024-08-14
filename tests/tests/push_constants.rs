use std::num::NonZeroU64;

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

/// We want to test that partial updates to push constants work as expected.
///
/// As such, we dispatch two compute passes, one which writes the values
/// before a partial update, and one which writes the values after the partial update.
///
/// If the update code is working correctly, the values not written to by the second update
/// will remain unchanged.
#[gpu_test]
static PARTIAL_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .limits(wgpu::Limits {
                max_push_constant_size: 32,
                ..Default::default()
            }),
    )
    .run_async(partial_update_test);

const SHADER: &str = r#"
    struct Pc {
        offset: u32,
        vector: vec4f,
    }

    var<push_constant> pc: Pc;

    @group(0) @binding(0)
    var<storage, read_write> output: array<vec4f>;

    @compute @workgroup_size(1)
    fn main() {
        output[pc.offset] = pc.vector;
    }
"#;

async fn partial_update_test(ctx: TestingContext) {
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..32,
            }],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // -- Dispatch 0 --

        // Dispatch number
        cpass.set_push_constants(0, bytemuck::bytes_of(&[0_u32]));
        // Update the whole vector.
        cpass.set_push_constants(16, bytemuck::bytes_of(&[1.0_f32, 2.0, 3.0, 4.0]));
        cpass.dispatch_workgroups(1, 1, 1);

        // -- Dispatch 1 --

        // Dispatch number
        cpass.set_push_constants(0, bytemuck::bytes_of(&[1_u32]));
        // Update just the y component of the vector.
        cpass.set_push_constants(20, bytemuck::bytes_of(&[5.0_f32]));
        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, 32);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let data = cpu_buffer.slice(..).get_mapped_range();

    let floats: &[f32] = bytemuck::cast_slice(&data);

    // first 4 floats the initial value
    // second 4 floats the first update
    assert_eq!(floats, [1.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 4.0]);
}
