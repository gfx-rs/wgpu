use wgpu::Backends;
use wgpu_test::{
    apply, fail, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        NO_TARGETLESS_RENDER,
        COMPUTE_PIPELINE_SPV_ENTRY_POINT_NAME_TOO_LONG,
    ]);
}

const INVALID_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("invalid shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("not valid wgsl")),
};

const TRIVIAL_COMPUTE_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial compute shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@compute @workgroup_size(1) fn main() {}",
    )),
};

const TRIVIAL_VERTEX_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial vertex shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }",
    )),
};

const TRIVIAL_FRAGMENT_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial fragment shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0); }",
    )),
};

// Create an invalid shader and a compute pipeline that uses it
// with a default bindgroup layout, and then ask for that layout.
// Validation should fail, but wgpu should not panic.
#[apply(gpu_test!)]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("doesn't exist"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    // https://github.com/gfx-rs/wgpu/issues/4167 this used to panic
                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[apply(gpu_test!)]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(TRIVIAL_COMPUTE_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[apply(gpu_test!)]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: wgpu::VertexState {
                                    module: &module,
                                    entry_point: Some("doesn't exist"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: None,
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[apply(gpu_test!)]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
                    let fs_module = ctx
                        .device
                        .create_shader_module(TRIVIAL_FRAGMENT_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: wgpu::VertexState {
                                    module: &vs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: Some(wgpu::FragmentState {
                                    module: &fs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    targets: &[Some(wgpu::ColorTargetState {
                                        format: wgpu::TextureFormat::Rgba8Unorm,
                                        blend: None,
                                        write_mask: wgpu::ColorWrites::ALL,
                                    })],
                                }),
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[apply(gpu_test!)]
static NO_TARGETLESS_RENDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        fail(
            &ctx.device,
            || {
                // Testing multisampling is important, because some backends don't behave well if one
                // tries to compile code in an unsupported multisample count. Failing to validate here
                // has historically resulted in requesting the back end to compile code.
                for power_of_two in [1, 2, 4, 8, 16, 32, 64] {
                    let _ = ctx
                        .device
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
                            multiview_mask: None,
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

/// An entry-point name too long for SPIR-V's `OpEntryPoint` must fail pipeline
/// creation with an internal error, not reach the driver as a malformed module.
#[apply(gpu_test!)]
static COMPUTE_PIPELINE_SPV_ENTRY_POINT_NAME_TOO_LONG: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                // Only the Vulkan backend generates SPIR-V.
                .skip(FailureCase::backend(Backends::all() - Backends::VULKAN)),
        )
        .run_async(|ctx| async move {
            // `OpEntryPoint` spends three words on its header, execution model, and
            // function id, so a nul-terminated name of this length makes the
            // instruction one word longer than SPIR-V can encode.
            let name = "a".repeat((usize::from(u16::MAX) - 3) * 4);
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        format!("@compute @workgroup_size(1) fn {name}() {{}}").into(),
                    ),
                });

            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);
            let _ = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some(&name),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let error = scope
                .pop()
                .await
                .expect("expected an internal error from pipeline creation");

            let message = error.to_string();
            assert!(
                message.contains("SPIR-V instructions are limited to 65535 words"),
                "unexpected error: {message}"
            );
        });
