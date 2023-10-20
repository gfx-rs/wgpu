use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static SYNC: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        // Taken from hello-compute tests.
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let ExecuteResults {
            patient_workgroup_results,
            hasty_workgroup_results: _,
        } = execute(&ctx.device, &ctx.queue, ARR_SIZE).await;
        assert_eq!(patient_workgroup_results, [16_u32; ARR_SIZE]);
    });
