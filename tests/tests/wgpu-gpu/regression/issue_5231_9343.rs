use wgpu::*;
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(READ_ONLY_DEPTH_WITHOUT_TEXTURE_BINDING);
    vec.push(READ_ONLY_DEPTH_WITH_SAMPLED_BINDING);
}

/// Regression tests for <https://github.com/gfx-rs/wgpu/issues/5231> and
/// <https://github.com/gfx-rs/wgpu/issues/9343>.
///
/// #5231 is a Vulkan validation error for a synchronization hazard (missing/incorrect
/// barrier) when using a read-only depth texture.
///
/// #9343 is a crash on dx12 when using a texture created without `TEXTURE_BINDING` as a
/// read-only depth attachment. With read-only depth, wgpu-core transitioned the depth
/// texture to `DEPTH_STENCIL_READ | RESOURCE`. This is normally a valid usage combination,
/// but when the texture view does not have `TEXTURE_BINDING` usage, the `RESOURCE` usage
/// is not allowed, and dx12 raises an error.
#[gpu_test]
static READ_ONLY_DEPTH_WITHOUT_TEXTURE_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
            .enable_noop()
            // <https://github.com/gfx-rs/wgpu/issues/5231>
            .expect_fail(wgpu_test::FailureCase {
                backends: Some(wgpu::Backends::VULKAN),
                reasons: vec![wgpu_test::FailureReason::validation_error()
                    .with_message("WRITE_AFTER_WRITE hazard detected.")],
                behavior: wgpu_test::FailureBehavior::AssertFailure,
                ..Default::default()
            }),
    )
    .run_sync(|ctx| read_only_depth_test(&ctx, false));

/// Test that a read-only depth attachment can simultaneously be sampled as a texture
/// binding within the same render pass. This exercises the `DEPTH_STENCIL_READ | RESOURCE`
/// usage combination that wgpu-core sets up in render.rs.
#[gpu_test]
static READ_ONLY_DEPTH_WITH_SAMPLED_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
            .enable_noop()
            // <https://github.com/gfx-rs/wgpu/issues/5231>
            .expect_fail(wgpu_test::FailureCase {
                backends: Some(wgpu::Backends::VULKAN),
                reasons: vec![wgpu_test::FailureReason::validation_error()
                    .with_message("WRITE_AFTER_WRITE hazard detected.")],
                behavior: wgpu_test::FailureBehavior::AssertFailure,
                ..Default::default()
            }),
    )
    .run_sync(|ctx| read_only_depth_test(&ctx, true));

/// Common implementation for read-only depth attachment tests.
///
/// When `sample_depth` is true, the depth texture is created with `TEXTURE_BINDING` and
/// the second (read-only) pass also binds it as a `texture_depth_2d` for sampling. This
/// exercises the `DEPTH_STENCIL_READ | RESOURCE` usage combination.
///
/// When `sample_depth` is false, the depth texture has only `RENDER_ATTACHMENT` usage and
/// the second pass only uses it as a read-only depth attachment.
fn read_only_depth_test(ctx: &TestingContext, sample_depth: bool) {
    let size = Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let color_texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&TextureViewDescriptor::default());

    let depth_usage = if sample_depth {
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING
    } else {
        TextureUsages::RENDER_ATTACHMENT
    };
    let depth_texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: depth_usage,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

    let simple_shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            concat!(
                "@vertex fn vs() -> @builtin(position) vec4f {\n",
                "    return vec4f(0.0, 0.0, 0.5, 1.0);\n",
                "}\n",
                "@fragment fn fs() -> @location(0) vec4f {\n",
                "    return vec4f(1.0);\n",
                "}\n",
            )
            .into(),
        ),
    });

    let empty_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });

    let color_target = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: None,
        write_mask: ColorWrites::all(),
    };

    let write_pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("depth write pipeline"),
            layout: Some(&empty_layout),
            vertex: VertexState {
                module: &simple_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &simple_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(color_target.clone())],
            }),
            multiview_mask: None,
            cache: None,
        });

    // For the read-only pass, optionally create a bind group and pipeline that
    // samples the depth texture.
    let bind_group;
    let readonly_layout;
    let sample_shader;

    let (readonly_pipeline_layout, readonly_module) = if sample_depth {
        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        bind_group = Some(ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depth_view),
            }],
        }));

        readonly_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        sample_shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(
                concat!(
                    "@group(0) @binding(0) var depth_tex: texture_depth_2d;\n",
                    "@vertex fn vs() -> @builtin(position) vec4f {\n",
                    "    return vec4f(0.0, 0.0, 0.5, 1.0);\n",
                    "}\n",
                    "@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n",
                    "    let d = textureLoad(depth_tex, vec2u(pos.xy), 0);\n",
                    "    return vec4f(d, 0.0, 0.0, 1.0);\n",
                    "}\n",
                )
                .into(),
            ),
        });

        (&readonly_layout, &sample_shader)
    } else {
        bind_group = None;
        (&empty_layout, &simple_shader)
    };

    let readonly_pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("depth read pipeline"),
            layout: Some(readonly_pipeline_layout),
            vertex: VertexState {
                module: readonly_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: None,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: readonly_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(color_target)],
            }),
            multiview_mask: None,
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    // First pass: writable depth, puts the depth texture in `DEPTH_STENCIL_WRITE` state.
    {
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("depth write pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&write_pipeline);
        rpass.draw(0..1, 0..1);
    }

    // Second pass: read-only depth attachment, triggers the `DEPTH_STENCIL_WRITE` ->
    // `DEPTH_STENCIL_READ` transition. When `sample_depth` is true, the depth texture
    // is also bound as a sampled texture, exercising `DEPTH_STENCIL_READ | RESOURCE`.
    {
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("depth read pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&readonly_pipeline);
        if let Some(bg) = &bind_group {
            rpass.set_bind_group(0, bg, &[]);
        }
        rpass.draw(0..1, 0..1);
    }

    ctx.queue.submit([encoder.finish()]);
}
