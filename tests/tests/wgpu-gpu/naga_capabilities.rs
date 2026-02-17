use wgpu_test::{GpuTestInitializer, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {}

pub fn validate_caps(ctx: TestingContext) {
    let device_caps = wgpu_core::device::features_to_naga_capabilities(
        ctx.adapter.features(),
        ctx.adapter.get_downlevel_capabilities().flags,
    );
    let max_caps = match ctx.adapter.get_info().backend {
        wgpu::Backend::Vulkan => naga::back::spv::supported_capabilities(),
        wgpu::Backend::Dx12 => naga::back::hlsl::supported_capabilities(),
        wgpu::Backend::Metal => naga::back::msl::supported_capabilities(),
        wgpu::Backend::Gl => naga::back::glsl::supported_capabilities(),
        wgpu::Backend::BrowserWebGpu => naga::back::wgsl::supported_capabilities(),
        wgpu::Backend::Noop => naga::valid::Capabilities::all(),
    };
    assert!(max_caps.contains(device_caps));
}
