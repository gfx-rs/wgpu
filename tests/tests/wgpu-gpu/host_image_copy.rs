use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(HOST_IMAGE_UPLOAD);
}

#[gpu_test]
static HOST_IMAGE_UPLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_sync(|ctx| {});
