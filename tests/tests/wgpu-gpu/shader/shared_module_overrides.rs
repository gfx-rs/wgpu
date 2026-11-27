use wgpu_test::{apply, gpu_test, GpuTestConfiguration, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(SHARED_MODULE_OVERRIDES);
}

// `value` has no default, so the vertex stage only builds if the fragment
// entry point is stripped before the overrides are processed.
const SHADER: &str = r#"
    override value: u32;

    @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
        return vec4f(vec2f(f32(vi & 1u), f32(vi >> 1u)) * 4.0 - 1.0, 0.0, 1.0);
    }

    @fragment fn fs_main() -> @location(0) u32 {
        return value;
    }
"#;

#[apply(gpu_test!)]
static SHARED_MODULE_OVERRIDES: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        for value in [7, 20] {
            render_and_check(&ctx, &module, value).await;
        }
    });

async fn render_and_check(ctx: &TestingContext, module: &wgpu::ShaderModule, value: u32) {
    let format = wgpu::TextureFormat::R32Uint;
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("value", f64::from(value))],
                    ..Default::default()
                },
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d::default(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture.create_view(&Default::default()),
                depth_slice: None,
                resolve_target: None,
                ops: Default::default(),
            })],
            ..Default::default()
        });
        rpass.set_pipeline(&pipeline);
        rpass.draw(0..3, 0..1);
    }
    let readback = wgpu_test::image::ReadbackBuffers::new(&ctx.device, &texture);
    readback.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit([encoder.finish()]);
    readback
        .assert_buffer_contents(ctx, &value.to_ne_bytes())
        .await;
}
