use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(RENDERBUFFER_TO_TEXTURE_ATTACHMENTS);
}

#[apply(gpu_test!)]
static RENDERBUFFER_TO_TEXTURE_ATTACHMENTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(renderbuffer_to_texture_attachments);

async fn renderbuffer_to_texture_attachments(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@vertex fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
                return vec4f(p[index], 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }"
                .into(),
        ),
    });
    let texture = |format, sample_count, usage| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    // Attachment-only images use renderbuffers on GLES; COPY_SRC images use textures.
    let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let ms_color = texture(wgpu::TextureFormat::Rgba8Unorm, 4, attachment);
    let ms_depth = texture(wgpu::TextureFormat::Depth24PlusStencil8, 4, attachment);
    let depth = texture(wgpu::TextureFormat::Depth24Plus, 1, attachment);
    let outputs = [
        texture(
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            attachment | wgpu::TextureUsages::COPY_SRC,
        ),
        texture(
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            attachment | wgpu::TextureUsages::COPY_SRC,
        ),
    ];
    let ms_color_view = ms_color.create_view(&Default::default());
    let ms_depth_view = ms_depth.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let output_views = outputs
        .each_ref()
        .map(|t| t.create_view(&Default::default()));
    let readbacks = outputs
        .each_ref()
        .map(|t| ReadbackBuffers::new(&ctx.device, t));
    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    // Reuse the draw framebuffer across multisampled, depth-only and depthless passes.
    for (sample_count, depth_format, color_view, depth_view) in [
        (
            4,
            Some(wgpu::TextureFormat::Depth24PlusStencil8),
            &ms_color_view,
            Some(&ms_depth_view),
        ),
        (
            1,
            Some(wgpu::TextureFormat::Depth24Plus),
            &output_views[0],
            Some(&depth_view),
        ),
        (1, None, &output_views[1], None),
    ] {
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_view.map(|view| {
                wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: (sample_count == 4).then_some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Discard,
                    }),
                }
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }

    for (readback, output) in readbacks.iter().zip(&outputs) {
        readback.copy_from(&ctx.device, &mut encoder, output);
    }
    ctx.queue.submit([encoder.finish()]);
    for readback in readbacks {
        readback
            .assert_buffer_contents(&ctx, &[255, 0, 0, 255].repeat(16))
            .await;
    }
}
