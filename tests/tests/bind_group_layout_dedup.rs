use wgpu_test::{gpu_test, GpuTestConfiguration, TestingContext};

#[gpu_test]
static BIND_GROUP_LAYOUT_DEDUPLICATION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(bgl_dedupe);

fn bgl_dedupe(ctx: TestingContext) {
    let entries_1 = &[];

    let entries_2 = &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];

    // Block so we can force all resource to die.
    {
        let bgl_1a = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_1,
            });

        let bgl_2 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_2,
            });

        let bgl_1b = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_1,
            });

        let bg_1a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl_1a,
            entries: &[],
        });

        let bg_1b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl_1b,
            entries: &[],
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl_1b],
                push_constant_ranges: &[],
            });

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

        let targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: Default::default(),
        })];

        let desc = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let pipeline = ctx.device.create_render_pipeline(&desc);

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&Default::default());

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: Default::default(),
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &bg_1b, &[]);

            pass.set_pipeline(&pipeline);

            pass.draw(0..6, 0..1);

            pass.set_bind_group(0, &bg_1a, &[]);

            pass.draw(0..6, 0..1);
        }

        ctx.queue.submit(Some(encoder.finish()));

        // Abuse the fact that global_id is really just the bitpacked ids when targeting wgpu-core.
        if ctx.adapter_info.backend != wgt::Backend::BrowserWebGpu {
            let bgl_1a_idx = bgl_1a.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_1a_idx, 0);
            let bgl_2_idx = bgl_2.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_2_idx, 1);
            let bgl_1b_idx = bgl_1b.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_1b_idx, 2);
        }
    }

    ctx.device.poll(wgpu::Maintain::Wait);

    if ctx.adapter_info.backend != wgt::Backend::BrowserWebGpu {
        // Now all of the BGL ids should be dead, so we should get the same ids again.
        for i in 0..=2 {
            let test_bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: entries_1,
                });

            let test_bgl_idx = test_bgl.global_id().inner() & 0xFFFF_FFFF;

            // https://github.com/gfx-rs/wgpu/issues/4912
            //
            // ID 2 is the deduplicated ID, which is never properly recycled.
            if i == 2 {
                assert_eq!(test_bgl_idx, 3);
            } else {
                assert_eq!(test_bgl_idx, i);
            }
        }
    }
}

const SHADER_SRC: &str = "
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
";
