use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(PRECOMPILE_ALL_STAGES_TEST);
}

#[gpu_test]
static PRECOMPILE_ALL_STAGES_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().features(wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS),
    )
    .run_async(async |ctx| unsafe {
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::precompile_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "vs_main"
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::precompile_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "fs_main"
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::precompile_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "cs_main"
            ));
    });
