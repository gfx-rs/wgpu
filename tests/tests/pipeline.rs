use wgpu_test::{fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

// Create an invalid shader and a compute pipeline that uses it
// with a default bindgroup layout, and then ask for that layout.
// Validation should fail, but wgpu should not panic.
#[gpu_test]
static PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // https://github.com/gfx-rs/wgpu/issues/4167
            .expect_fail(FailureCase::always().panic("Pipeline is invalid")),
    )
    .run_sync(|ctx| {
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        fail(&ctx.device, || {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl("not valid wgsl".into()),
                });

            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("mandelbrot compute pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: "doesn't exist",
                    constants: &Default::default(),
                });

            pipeline.get_bind_group_layout(0);
        });
    });
