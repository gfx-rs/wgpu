use wgpu::include_wgsl;

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static SHADER_COMPILE_SUCCESS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
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
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let sm = ctx
            .device
            .create_shader_module(include_wgsl!("error_shader.wgsl"));
        assert!(pollster::block_on(ctx.device.pop_error_scope()).is_some());

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
