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
            .create_shader_module_passthrough(wgpu::include_precompiled_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "vs_main",
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::include_precompiled_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "fs_main",
                all
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::include_precompiled_wgsl!(
                "tests/wgpu-gpu/precompile/shader.wgsl",
                "cs_main",
                glsl spirv wgsl hlsl msl
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::precompile_wgsl!(
                r#"
@compute
@workgroup_size(1)
fn cs_main() {}
        "#,
                "cs_main",
            ));
        // This is just the GLSL file compiled with glslang -V shader.vert -o shader.spv.
        // The spirv file must exist before parsing begins. I didn't want to add it to
        // the build script but that is another viable option.
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::include_precompiled_spirv!(
                "tests/wgpu-gpu/precompile/shader.spv",
                "main",
            ));
        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::include_precompiled_glsl!(
                "tests/wgpu-gpu/precompile/shader.vert",
                vertex,
            ));

        let _ = ctx
            .device
            .create_shader_module_passthrough(wgpu::precompile_glsl!(
                r#"
#version 450
const float c_scale = 1.2;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

void main() {
  v_uv = a_uv;
  gl_Position = vec4(c_scale * a_pos, 0.0, 1.0);
}
                "#,
                vertex,
            ));
    });
