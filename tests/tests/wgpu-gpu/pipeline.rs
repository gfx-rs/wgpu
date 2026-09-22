use wgpu_test::{apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        NO_TARGETLESS_RENDER,
        RENDER_PIPELINE_ASYNC_RESOLVES_A_USABLE_PIPELINE,
        RENDER_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_EXACTLY_ONCE,
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

// A pipeline awaited from `create_render_pipeline_async` must be usable for drawing: on the
// backends without an asynchronous form the future is already resolved, and on WebGPU it
// resolves with the pipeline `createRenderPipelineAsync()` compiled.
#[apply(gpu_test!)]
static RENDER_PIPELINE_ASYNC_RESOLVES_A_USABLE_PIPELINE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_async(|ctx| async move {
            let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
            let fs_module = ctx
                .device
                .create_shader_module(TRIVIAL_FRAGMENT_SHADER_DESC);

            let pipeline = ctx
                .device
                .create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
                    label: Some("async render pipeline"),
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
                })
                .await
                .expect("async render pipeline creation failed");

            let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("async render pipeline target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("async render pipeline pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations::default(),
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&pipeline);
                pass.draw(0..3, 0..1);
            }
            ctx.queue.submit([encoder.finish()]);

            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
        });

// An invalid descriptor has to be reported exactly once: WebGPU rejects the promise and
// leaves the error scope empty, every other backend creates the pipeline synchronously and
// reports through the error scope while the future resolves to `Ok`. Reporting through
// both would make the caller handle the same failure twice; reporting through neither would
// lose it.
#[apply(gpu_test!)]
static RENDER_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_EXACTLY_ONCE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_async(|ctx| async move {
            let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);

            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            // Neither color targets nor a depth-stencil attachment: the pipeline has
            // nothing to render to.
            let pipeline =
                ctx.device
                    .create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
                        label: Some("targetless async render pipeline"),
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
                        fragment: None,
                        multiview_mask: None,
                        cache: None,
                    });
            // Pop before awaiting: the guard is neither `Send` nor `Sync`, and a synchronous
            // backend has already reported into the scope by now.
            let scope_result = scope.pop();

            let from_future = pipeline.await.err();
            let from_scope = scope_result.await;

            assert_ne!(
                from_future.is_some(),
                from_scope.is_some(),
                "expected exactly one report of the invalid descriptor, \
                 got future error {from_future:?} and scope error {from_scope:?}"
            );
        });
