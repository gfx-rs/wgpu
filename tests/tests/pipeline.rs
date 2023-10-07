use wasm_bindgen_test::*;
use wgpu_test::{fail, initialize_test, FailureCase, TestParameters};

#[test]
#[wasm_bindgen_test]
fn pipeline_default_layout_bad_module() {
    // Create an invalid shader and a compute pipeline that uses it
    // with a default bindgroup layout, and then ask for that layout.
    // Validation should fail, but wgpu should not panic.
    let parameters = TestParameters::default()
        .skip(FailureCase::webgl2())
        // https://github.com/gfx-rs/wgpu/issues/4167
        .expect_fail(FailureCase::always());
    initialize_test(parameters, |ctx| {
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
                });

            pipeline.get_bind_group_layout(0);
        });
    });
}
