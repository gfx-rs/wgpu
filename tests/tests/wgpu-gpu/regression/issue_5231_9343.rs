use wgpu::*;
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(READ_ONLY_DEPTH_WITHOUT_TEXTURE_BINDING);
}

/// Regression test for <https://github.com/gfx-rs/wgpu/issues/9343>, a dx12 crash when
/// using a texture created without `TEXTURE_BINDING` as a read-only depth attachment.
///
/// When both depth and stencil were read-only, wgpu-core transitioned the depth
/// texture to `DEPTH_STENCIL_READ | RESOURCE`. The `RESOURCE` usage maps to
/// `D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE` on
/// dx12. However, a depth texture created without `TEXTURE_BINDING` cannot
/// be transitioned to these states on dx12.
#[gpu_test]
static READ_ONLY_DEPTH_WITHOUT_TEXTURE_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
            .enable_noop(),
    )
    .run_sync(|ctx| {
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

        // Depth texture with _only_ `RENDER_ATTACHMENT` (no `TEXTURE_BINDING`).
        let depth_texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let vertex = VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        };

        let fragment = FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::all(),
            })],
        };

        // Write pipeline: depth_write_enabled = true
        let write_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("depth write pipeline"),
                layout: Some(&pipeline_layout),
                vertex: vertex.clone(),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::Always),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(fragment.clone()),
                multiview_mask: None,
                cache: None,
            });

        // Read-only pipeline: depth_write_enabled = false
        let readonly_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("depth read pipeline"),
                layout: Some(&pipeline_layout),
                vertex,
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: None,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(fragment),
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

        // Second pass: read-only depth, triggers the `DEPTH_STENCIL_WRITE` ->
        // `DEPTH_STENCIL_READ` transition. Before the fix, this would include
        // `RESOURCE` usage, even though the texture does not have
        // `TEXTURE_BINDING`
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
                    depth_ops: None,   // read-only depth
                    stencil_ops: None, // read-only stencil
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&readonly_pipeline);
            rpass.draw(0..1, 0..1);
        }

        ctx.queue.submit([encoder.finish()]);
    });
