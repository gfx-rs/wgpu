use wgpu::include_wgsl;

use wgpu_test::{fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        SHADER_COMPILE_SUCCESS,
        SHADER_COMPILE_ERROR,
        ENABLE_EXTENSION_AVAILABLE,
        ENABLE_EXTENSION_UNAVAILABLE,
    ]);
}

#[gpu_test]
static SHADER_COMPILE_SUCCESS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let sm = ctx
            .device
            .create_shader_module(include_wgsl!("successful_shader.wgsl"));

        let compilation_info = sm.get_compilation_info().await;
        for message in compilation_info.messages.iter() {
            assert!(message.message_type != wgpu::CompilationMessageType::Error);
        }
    });

#[gpu_test]
static SHADER_COMPILE_ERROR: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let sm = ctx
            .device
            .create_shader_module(include_wgsl!("error_shader.wgsl"));
        assert!(pollster::block_on(scope.pop()).is_some());

        let compilation_info = sm.get_compilation_info().await;
        let error_message = compilation_info
            .messages
            .iter()
            .find(|message| message.message_type == wgpu::CompilationMessageType::Error)
            .expect("Expected error message not found");
        let span = error_message.location.expect("Expected span not found");
        assert_eq!(
            span.offset, 32,
            "Expected the offset to be 32, because we're counting UTF-8 bytes"
        );
        assert_eq!(span.length, 1, "Expected length to roughly be 1"); // Could be relaxed, depending on the parser requirements.
        assert_eq!(
            span.line_number, 1,
            "Expected the line number to be 1, because we're counting lines from 1"
        );
        assert_eq!(
            span.line_position, 33,
            "Expected the column number to be 33, because we're counting lines from 1"
        );
    });

const ENABLE_EXTENSION_SHADER_SOURCE: &str = r#"
    enable f16;

    @compute @workgroup_size(1)
    fn main() {}
"#;

#[gpu_test]
static ENABLE_EXTENSION_AVAILABLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::SHADER_F16)
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        valid(&ctx.device, || {
            let _ = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("shader declaring enable extension"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                        ENABLE_EXTENSION_SHADER_SOURCE,
                    )),
                });
        });
    });

#[gpu_test]
static ENABLE_EXTENSION_UNAVAILABLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // SHADER_F16 feature not requested
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        fail(
            &ctx.device,
            || {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("shader declaring enable extension"),
                        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                            ENABLE_EXTENSION_SHADER_SOURCE,
                        )),
                    })
            },
            Some("the `f16` extension is not supported in the current environment"),
        );
    });
