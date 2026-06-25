use std::borrow::Cow;

use wgpu_test::{fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        DEBUG_PRINTF_SHADER_MODULE,
        DEBUG_PRINTF_REQUIRES_FEATURE,
        DEBUG_PRINTF_REQUIRES_ENABLE_EXTENSION,
        DEBUG_PRINTF_REJECTS_STRING_LITERAL_OUTSIDE_CALL,
    ]);
}

const DEBUG_PRINTF_SHADER: &str = r#"
enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    debugPrintf("debug value: %d", 1i);
}
"#;

const DEBUG_PRINTF_WITHOUT_ENABLE_SHADER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    debugPrintf("debug value: %d", 1i);
}
"#;

const STRING_LITERAL_OUTSIDE_DEBUG_PRINTF_SHADER: &str = r#"
enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    let _value = "not a debugPrintf format";
}
"#;

#[gpu_test]
static DEBUG_PRINTF_SHADER_MODULE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .enable_noop()
            .features(wgpu::Features::DEBUG_PRINTF)
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        valid(&ctx.device, || {
            create_shader_module(&ctx.device, DEBUG_PRINTF_SHADER);
        });
    });

#[gpu_test]
static DEBUG_PRINTF_REQUIRES_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .enable_noop()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        fail(
            &ctx.device,
            || create_shader_module(&ctx.device, DEBUG_PRINTF_SHADER),
            Some("DEBUG_PRINTF"),
        );
    });

#[gpu_test]
static DEBUG_PRINTF_REQUIRES_ENABLE_EXTENSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .enable_noop()
            .features(wgpu::Features::DEBUG_PRINTF)
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        fail(
            &ctx.device,
            || create_shader_module(&ctx.device, DEBUG_PRINTF_WITHOUT_ENABLE_SHADER),
            Some("enable extension is not enabled"),
        );
    });

#[gpu_test]
static DEBUG_PRINTF_REJECTS_STRING_LITERAL_OUTSIDE_CALL: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .enable_noop()
                .features(wgpu::Features::DEBUG_PRINTF)
                .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                .limits(wgpu::Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            fail(
                &ctx.device,
                || create_shader_module(&ctx.device, STRING_LITERAL_OUTSIDE_DEBUG_PRINTF_SHADER),
                Some("String literals are only supported in debugPrintf"),
            );
        });

fn create_shader_module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("debugPrintf shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}
