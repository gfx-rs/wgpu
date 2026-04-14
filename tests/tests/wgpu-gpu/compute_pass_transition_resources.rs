use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_PASS_TRANSITION_RESOURCES);
}

#[gpu_test]
static COMPUTE_PASS_TRANSITION_RESOURCES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 128,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.transition_resources(
            [wgpu::BufferTransition {
                buffer: &buffer,
                state: wgpu::BufferUses::STORAGE_READ_WRITE,
            }]
            .into_iter(),
            core::iter::empty(),
        );

        drop(pass);

        ctx.queue.submit([encoder.finish()]);
    });
