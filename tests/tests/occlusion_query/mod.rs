use std::mem::size_of;
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static OCCLUSION_QUERY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
    .run_async(|ctx| async move {
        // Create depth texture
        let depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth texture"),
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
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Setup pipeline using a simple shader with hardcoded vertices
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: None,
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
                cache: None,
            });

        // Create occlusion query set
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Query set"),
            ty: wgpu::QueryType::Occlusion,
            count: 3,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: Some(&query_set),
            });
            render_pass.set_pipeline(&pipeline);

            // Not occluded (z = 1.0, nothing drawn yet)
            render_pass.begin_occlusion_query(0);
            render_pass.draw(4..7, 0..1);
            render_pass.end_occlusion_query();

            // Not occluded (z = 0.0)
            render_pass.begin_occlusion_query(1);
            render_pass.draw(0..3, 0..1);
            render_pass.end_occlusion_query();

            // Occluded (z = 1.0)
            render_pass.begin_occlusion_query(2);
            render_pass.draw(4..7, 0..1);
            render_pass.end_occlusion_query();
        }

        // Resolve query set to buffer
        let query_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query buffer"),
            size: size_of::<u64>() as u64 * 3,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(&query_set, 0..3, &query_buffer, 0);

        let mapping_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mapping buffer"),
            size: query_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&query_buffer, 0, &mapping_buffer, 0, query_buffer.size());

        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        let query_buffer_view = mapping_buffer.slice(..).get_mapped_range();
        let query_data: &[u64; 3] = bytemuck::from_bytes(&query_buffer_view);

        // WebGPU only defines query results as zero/non-zero
        assert_ne!(query_data[0], 0);
        assert_ne!(query_data[1], 0);
        assert_eq!(query_data[2], 0);
    });
