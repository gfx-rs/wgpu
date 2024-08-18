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

        fail(
            &ctx.device,
            || {
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl("not valid wgsl".into()),
                    });

                let pipeline =
                    ctx.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("mandelbrot compute pipeline"),
                            layout: None,
                            module: &module,
                            entry_point: Some("doesn't exist"),
                            compilation_options: Default::default(),
                            cache: None,
                        });

                pipeline.get_bind_group_layout(0);
            },
            None,
        );
    });

const TRIVIAL_VERTEX_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial vertex shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }",
    )),
};

#[gpu_test]
static NO_TARGETLESS_RENDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        fail(
            &ctx.device,
            || {
                // Testing multisampling is important, because some backends don't behave well if one
                // tries to compile code in an unsupported multisample count. Failing to validate here
                // has historically resulted in requesting the back end to compile code.
                for power_of_two in [1, 2, 4, 8, 16, 32, 64] {
                    ctx.device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: None,
                            layout: None,
                            vertex: wgpu::VertexState {
                                module: &ctx
                                    .device
                                    .create_shader_module(TRIVIAL_VERTEX_SHADER_DESC),
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                buffers: &[],
                            },
                            primitive: Default::default(),
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState {
                                count: power_of_two,
                                ..Default::default()
                            },
                            fragment: None,
                            multiview: None,
                            cache: None,
                        });
                }
            },
            Some(concat!(
                "At least one color attachment or depth-stencil attachment was expected, ",
                "but no render target for the pipeline was specified."
            )),
        )
    });
