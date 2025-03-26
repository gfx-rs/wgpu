use wgpu_test::{gpu_test, GpuTestConfiguration};

#[gpu_test]
static INITIALIZE: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|_ctx| {});
