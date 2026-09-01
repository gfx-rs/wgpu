use wgpu_test::{
    apply, fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        COMPUTE_PIPELINE_QUERY_SUBGROUP_SIZE_SIMPLE,
        COMPUTE_PIPELINE_QUERY_SUBGROUP_SIZE_VERIFY,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        NO_TARGETLESS_RENDER,
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

// Simply test that `ComputePipeline::get_subgroup_size` returns `Some(..)` on Metal and `None` otherwise.
// For that we need a basic compute pipeline which makes up most of the code below.
#[apply(gpu_test!)]
static COMPUTE_PIPELINE_QUERY_SUBGROUP_SIZE_SIMPLE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(|ctx| {
            valid(&ctx.device, || {
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

                if ctx.adapter.get_info().backend == wgpu::Backend::Metal {
                    assert!(pipeline.get_subgroup_size().is_some())
                } else {
                    assert!(pipeline.get_subgroup_size().is_none())
                }
            });
        });

// Verify that `ComputePipeline::get_subgroup_size` returns the same value we get from the builtin `subgroup_size`.
// For this we need "the whole thing":
// setup and run a compute pass where we write the value of the builtin to a buffer, copy, readback, and compare.
#[apply(gpu_test!)]
static COMPUTE_PIPELINE_QUERY_SUBGROUP_SIZE_VERIFY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(wgpu::Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            if ctx.adapter.get_info().backend != wgpu::Backend::Metal {
                return; // only works on metal
            }

            let (device, queue) = ctx
                .adapter
                .request_device(&Default::default())
                .await
                .unwrap();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let shader = format!(
                "
            @group(0) @binding(0) var<storage, read_write> out: u32;
            @compute @workgroup_size(1)
            fn main(@builtin(subgroup_size) subgroup_size: u32) {{
                out = subgroup_size;
            }}"
            );

            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pl),
                entry_point: Some("main"),
                compilation_options: Default::default(),
                module: &module,
                cache: None,
            });

            let subgroup_size = pipeline
                .get_subgroup_size()
                .expect("We should get something on Metal");

            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(out.as_entire_buffer_binding()),
                }],
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_bind_group(0, &bg, &[]);
                pass.set_pipeline(&pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, 4);
            queue.submit(Some(encoder.finish()));

            readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            assert_eq!(
                &*readback.slice(..).get_mapped_range().unwrap(),
                &subgroup_size.to_le_bytes()
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
