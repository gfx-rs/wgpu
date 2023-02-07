use crate::common::{initialize_test, TestParameters};

use wasm_bindgen_test::wasm_bindgen_test;
use wgpu::*;

#[wasm_bindgen_test]
#[test]
fn pass_reset_vertex_buffer() {
    initialize_test(TestParameters::default(), |ctx| {
        let module = ctx
            .device
            .create_shader_module(include_wgsl!("issue_3457.wgsl"));

        let vertex_buffer1 = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("vertex buffer 1"),
            size: 6 * 16,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let vertex_buffer2 = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("vertex buffer 2"),
            size: 6 * 4,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let double_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Double Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: "double_buffer_vert",
                    buffers: &[
                        VertexBufferLayout {
                            array_stride: 16,
                            step_mode: VertexStepMode::Vertex,
                            attributes: &vertex_attr_array![0 => Float32x4],
                        },
                        VertexBufferLayout {
                            array_stride: 4,
                            step_mode: VertexStepMode::Vertex,
                            attributes: &vertex_attr_array![1 => Float32],
                        },
                    ],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: "double_buffer_frag",
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::all(),
                    })],
                }),
                multiview: None,
            });

        let single_pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Single Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: "single_buffer_vert",
                    buffers: &[VertexBufferLayout {
                        array_stride: 16,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![0 => Float32x4],
                    }],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: None,
                multiview: None,
            });

        let view = ctx
            .device
            .create_texture(&TextureDescriptor {
                label: Some("Render texture"),
                size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());

        let mut encoder1 = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut double_rpass = encoder1.begin_render_pass(&RenderPassDescriptor {
            label: Some("double renderpass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: false,
                },
            })],
            depth_stencil_attachment: None,
        });

        double_rpass.set_pipeline(&double_pipeline);
        double_rpass.set_vertex_buffer(0, vertex_buffer1.slice(..));
        double_rpass.set_vertex_buffer(1, vertex_buffer2.slice(..));
        double_rpass.draw(0..3, 0..1);

        drop(double_rpass);

        ctx.queue.submit(Some(encoder1.finish()));

        drop((vertex_buffer2, view));

        ctx.device.poll(Maintain::Wait);

        let mut encoder2 = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let depth_view = ctx
            .device
            .create_texture(&TextureDescriptor {
                label: Some("Depth texture"),
                size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());

        let mut single_rpass = encoder2.begin_render_pass(&RenderPassDescriptor {
            label: Some("single renderpass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: false,
                }),
                stencil_ops: None,
            }),
        });

        single_rpass.set_pipeline(&single_pipeline);
        single_rpass.set_vertex_buffer(0, vertex_buffer1.slice(..));
        single_rpass.draw(0..6, 0..1);

        drop(single_rpass);

        ctx.queue.submit(Some(encoder2.finish()));
    })
}
