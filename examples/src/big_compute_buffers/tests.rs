use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static COMPUTE_1: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::BUFFER_BINDING_ARRAY
                    | Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            )
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS),
    )
    .run_async(|ctx| {
        // We assume the test GPU in the testing environment's maximum supported buffer size
        // is < 256MB.
        const SIZE: usize = (256 << 20) / std::mem::size_of::<f32>(); //256mb of f32s
        let input = &[0.0; SIZE];

        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input).await }
    });

async fn assert_execute_gpu(device: &wgpu::Device, queue: &wgpu::Queue, input: &[f32]) {
    let expected_len = input.len();
    if let Some(produced) = execute_gpu_inner(device, queue, input).await {
        assert_eq!(produced.len(), expected_len);
        assert!(produced.into_iter().all(|v| v == 1.0));
    }
}
