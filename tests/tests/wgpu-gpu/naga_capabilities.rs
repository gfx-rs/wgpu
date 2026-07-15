use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(VALIDATE_CAPABILITIES);
}

pub fn validate_capabilities(ctx: TestingContext) {
    use naga::valid::Capabilities as Caps;
    let device_caps = wgpu_core::device::features_to_naga_capabilities(
        ctx.adapter.features(),
        ctx.adapter.get_downlevel_capabilities().flags,
    );
    let max_caps = match ctx.adapter.get_info().backend {
        wgpu::Backend::Vulkan => naga::back::spv::supported_capabilities(),
        wgpu::Backend::Dx12 => naga::back::hlsl::supported_capabilities(),
        wgpu::Backend::Metal => {
            naga::back::msl::supported_capabilities()
                // TODO(https://github.com/gfx-rs/wgpu/issues/9849): mask off
                // BUFFER_BINDING_ARRAY because it is not currently reported by
                // naga's MSL backend.
                | Caps::BUFFER_BINDING_ARRAY
        }
        wgpu::Backend::Gl => naga::back::glsl::supported_capabilities(),
        wgpu::Backend::BrowserWebGpu => naga::back::wgsl::supported_capabilities(),
        wgpu::Backend::Noop => Caps::all(),
    };
    let diff = device_caps - max_caps;
    assert_eq!(diff, Caps::empty());
}

#[gpu_test]
static VALIDATE_CAPABILITIES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::empty())
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_sync(validate_capabilities);
