use crate::common::{initialize_test, TestParameters};
use std::borrow::Cow;
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
#[wasm_bindgen_test]
fn occlusion_query() {
    initialize_test(
        TestParameters::default().downlevel_flags(wgpu::DownlevelFlags::OCCLUSION_QUERY),
        |ctx| {
            // Create albedo texture
            let color_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let color_texture_view =
                color_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create depth texture
            let depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Setup pipeline with simple shader with hardcoded vertices
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[],
                        push_constant_ranges: &[],
                    });
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
                });
            let pipeline = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(color_texture.format().into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                });

            // Create occlusion query set
            let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Occlusion,
                count: 3,
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_texture_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: Some(&query_set),
                });
                render_pass.set_pipeline(&pipeline);

                // Not occluded (nothing drawn yet)
                render_pass.begin_occlusion_query(0);
                render_pass.draw(4..7, 0..1);
                render_pass.end_occlusion_query();

                // Not occluded (z = 0.0)
                render_pass.begin_occlusion_query(1);
                render_pass.draw(0..3, 0..1);
                render_pass.end_occlusion_query();

                // Occluded (z = 0.5)
                render_pass.begin_occlusion_query(2);
                render_pass.draw(4..7, 0..1);
                render_pass.end_occlusion_query();
            }

            // Resolve query set to buffer
            let query_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<u64>() as u64 * 3,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(&query_set, 0..3, &query_buffer, 0);

            ctx.queue.submit(Some(encoder.finish()));

            query_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            ctx.device.poll(wgpu::Maintain::Wait);
            let query_buffer_view = query_buffer.slice(..).get_mapped_range();
            let query_data: &[u64; 3] = bytemuck::from_bytes(&query_buffer_view);

            assert_eq!(query_data[0], 2048);
            assert_eq!(query_data[1], 2048);
            assert_eq!(query_data[2], 0);
        },
    )
}
