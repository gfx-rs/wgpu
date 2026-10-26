use wgpu::util::DeviceExt;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_WORKGROUP_LIMITS);
}

#[apply(gpu_test!)]
static COMPUTE_WORKGROUP_LIMITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .instance_flags(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL),
    )
    .run_async(|ctx| async move {
        let maximum = ctx
            .device
            .limits()
            .max_compute_invocations_per_workgroup
            .min(ctx.device.limits().max_compute_workgroup_size_x);
        let indirect = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[1u32, 1, 1]),
                usage: wgpu::BufferUsages::INDIRECT,
            });
        for [x, y, z] in [[1, 1, 1], [32, 1, 1], [4, 2, 2], [maximum, 1, 1]] {
            let count = x * y * z;
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        format!(
                            "@group(0) @binding(0) var<storage, read_write> values: array<u32>;
                     @compute @workgroup_size({x}, {y}, {z})
                     fn main(@builtin(local_invocation_index) index: u32) {{
                         values[index] = index + 1u;
                     }}"
                        )
                        .into(),
                    ),
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            for indirect_dispatch in [false, true] {
                let size = u64::from(count) * 4;
                let output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    }],
                });
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    if indirect_dispatch {
                        pass.dispatch_workgroups_indirect(&indirect, 0);
                    } else {
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
                ctx.queue.submit([encoder.finish()]);
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, Result::unwrap);
                ctx.async_poll(wgpu::PollType::wait_indefinitely())
                    .await
                    .unwrap();
                let mapped = readback.slice(..).get_mapped_range().unwrap();
                assert_eq!(
                    bytemuck::cast_slice::<u8, u32>(&mapped),
                    (1..=count).collect::<Vec<_>>(),
                    "workgroup {x}x{y}x{z}, indirect={indirect_dispatch}",
                );
            }
        }
    });
