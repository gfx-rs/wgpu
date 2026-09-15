use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use wgpu_test::{
    apply, capture_logs, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(DEBUG_PRINTF_OUTPUT);
}

#[apply(gpu_test!)]
static DEBUG_PRINTF_OUTPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::DEBUG_PRINTF)
            .limits(wgpu::Limits::downlevel_defaults())
            .instance_flags(wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG_PRINTF)
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS),
    )
    .run_sync(|ctx| {
        let logs = capture_logs();
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("debugPrintf output"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
enable wgpu_debug_printf;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    debugPrintf("wgpu-debug-printf-test: invocation=%u value=%u", id.x, id.x * 3u + 7u);
}
"#
                    .into(),
                ),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("debugPrintf output"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let submission = ctx.queue.submit([encoder.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(10)),
            })
            .unwrap();

        let mut remaining: BTreeSet<_> = (0..4)
            .map(|id| format!("invocation={id} value={}", id * 3 + 7))
            .collect();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !remaining.is_empty() {
            let (level, message) = logs
                .messages
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| panic!("Missing shader messages {remaining:?}: {error}"));
            if let Some((_, value)) = message.split_once("wgpu-debug-printf-test: ") {
                assert_eq!(level, log::Level::Info);
                assert!(message.contains("[shader debugPrintf]"), "{message}");
                assert!(
                    remaining.remove(value.trim()),
                    "Unexpected shader message: {message}"
                );
            }
        }
    });
