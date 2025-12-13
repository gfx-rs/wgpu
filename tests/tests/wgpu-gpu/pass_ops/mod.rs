use wgpu_test::{
    gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DONT_CARE);
}

#[gpu_test]
static DONT_CARE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(run_test);

async fn run_test(ctx: TestingContext) {
    let shader_src = "
        const triangles = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
            return vec4f(triangles[vertex_index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(127.0 / 255.0);
        }
    ";

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            buffers: &[],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readbacks = ReadbackBuffers::new(&ctx.device, &out_texture);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &out_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() }),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    rpass.set_pipeline(&pipeline);
    rpass.draw(0..3, 0..1);

    drop(rpass);

    readbacks.copy_from(&ctx.device, &mut encoder, &out_texture);

    ctx.queue.submit([encoder.finish()]);

    // Assert that DONT_CARE load op was fully overridden by the draw.
    readbacks
        .assert_buffer_contents(&ctx, &[127, 127, 127, 127])
        .await;
}
