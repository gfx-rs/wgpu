use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(INITIALIZE);
}

#[gpu_test]
static INITIALIZE: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|_ctx| {});
