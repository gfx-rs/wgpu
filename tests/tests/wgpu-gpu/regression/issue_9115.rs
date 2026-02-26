use wgpu::util::DeviceExt;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(IMMEDIATES_WITH_UNIFORM_IN_SINGLE_MODULE);
}

/// On the GLES backend, using immediates in both vertex and fragment shaders from a single
/// shader module, while also having a uniform buffer with a struct type, caused a panic.
/// The backend incorrectly encountered the uniform's struct type when processing immediates.
///
/// See <https://github.com/gfx-rs/wgpu/issues/9115>.
#[gpu_test]
static IMMEDIATES_WITH_UNIFORM_IN_SINGLE_MODULE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::IMMEDIATES)
            .limits(wgpu::Limits {
                max_immediate_size: 32,
                ..Default::default()
            }),
    )
    .run_async(immediates_with_uniform_in_single_module);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Immediates {
    position: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
}

async fn immediates_with_uniform_in_single_module(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_9115.wgsl"));

    let globals_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::cast_slice(&[1.0_f32, 1.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });

    let pll = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pll"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: size_of::<Immediates>() as u32,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pll),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_immediates(
            0,
            bytemuck::cast_slice(&[Immediates {
                position: [0.0, 0.0],
                size: [1.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            }]),
        );
        rpass.draw(0..3, 0..1);
    }

    let buffers = ReadbackBuffers::new(&ctx.device, &texture);
    buffers.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit([encoder.finish()]);

    // The fragment shader outputs immediates.color, which is green (0, 1, 0, 1).
    let expected: [u8; 4] = [0, 255, 0, 255];
    buffers.assert_buffer_contents(&ctx, &expected).await;
}
