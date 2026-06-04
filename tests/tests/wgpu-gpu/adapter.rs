use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(STRICT_WEBGPU_COMPLIANCE_ADAPTER);
}

#[gpu_test]
static STRICT_WEBGPU_COMPLIANCE_ADAPTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        assert!(ctx
            .adapter
            .get_downlevel_capabilities()
            .is_webgpu_compliant());
        assert!(wgpu::Limits::defaults().check_limits(&ctx.adapter.limits()));
    });
