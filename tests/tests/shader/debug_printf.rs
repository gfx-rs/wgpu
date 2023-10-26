use wgpu::{
    include_wgsl, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    Features, Limits, Maintain, PipelineLayoutDescriptor,
};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static DEBUG_PRINTF: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::DEBUG_PRINTF)
            .limits(Limits::default()),
    )
    .run_sync(|ctx| {
        let pll = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let sm = ctx
            .device
            .create_shader_module(include_wgsl!("debug_printf.wgsl"));

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("debugprintf"),
                layout: Some(&pll),
                module: &sm,
                entry_point: "main",
            });

        // -- Run test --

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
        }

        ctx.queue.submit(Some(encoder.finish()));

        ctx.device.poll(Maintain::Wait);
    });
