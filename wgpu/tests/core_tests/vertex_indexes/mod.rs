use std::num::NonZeroU64;

use crate::core_tests::common::init::{initialize_test, TestParameters};

#[test]
fn draw() {
    initialize_test(
        TestParameters::default().features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
        |ctx| {
            let shader = ctx
                .device
                .create_shader_module(&wgpu::include_wgsl!("draw.vert.wgsl"));

            let bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        visibility: wgpu::ShaderStage::VERTEX,
                        count: None,
                    }],
                });

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4 * 6,
                usage: wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            });

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            let ppl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[]
            });

            let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&ppl),
                vertex: wgpu::VertexState {
                    buffers: &[],
                    entry_point: "vs_main",
                    module: &shader,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: None,
            });

            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor::default());

            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..6, 0..1);

            drop(rpass);

            ctx.queue.submit(Some(encoder.finish()));
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read);
            ctx.device.poll(wgpu::Maintain::Wait);
            let data: Vec<u32> = bytemuck::cast_slice(&*slice.get_mapped_range()).to_vec();

            assert_eq!(data, [0, 1, 2, 3, 4, 5]);
        },
    )
}
