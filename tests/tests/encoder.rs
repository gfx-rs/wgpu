use wgpu_test::{gpu_test, infra::GpuTestConfiguration};

#[gpu_test]
static DROP_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    drop(encoder);
});
