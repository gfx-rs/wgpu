use crate::common::{initialize_test, TestParameters};

use wasm_bindgen_test::wasm_bindgen_test;
use wgpu::*;

/// The core issue here was that we weren't properly disabling vertex attributes on GL
/// when a renderpass ends. This ended up being rather tricky to test for as GL is remarkably
/// tolerant of errors. This test, with the fix not-applied, only fails on WebGL.
///
/// We need to setup a situation where it's invalid to issue a draw call without the fix.
/// To do this we first make a renderpass using two vertex buffers and draw on it. Then we
/// submit, delete the second vertex buffer and `poll(Wait)`. Because we maintained the device,
/// the actual underlying buffer for the second vertex buffer is deleted, causing a draw call
/// that is invalid if the second attribute is still enabled.
///
/// We use non-consecutive vertex attribute locations (0 and 5) in order to also test
/// that we unset the correct locations (see PR #3706).
#[wasm_bindgen_test]
#[test]
fn pass_reset_vertex_buffer() {
    initialize_test(TestParameters::default(), |ctx| {
        let module = ctx
            .device
            .create_shader_module(include_wgsl!("issue_3457.wgsl"));

        // We use two separate vertex buffers so we can delete one in between submisions
        let vertex_buffer1 = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("vertex buffer 1"),
            size: 3 * 16,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let vertex_buffer2 = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("vertex buffer 2"),
            size: 3 * 4,
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
                            attributes: &vertex_attr_array![5 => Float32],
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
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: "single_buffer_frag",
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::all(),
                    })],
                }),
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
            timestamp_writes: &[],
        });

        double_rpass.set_pipeline(&double_pipeline);
        double_rpass.set_vertex_buffer(0, vertex_buffer1.slice(..));
        double_rpass.set_vertex_buffer(1, vertex_buffer2.slice(..));
        double_rpass.draw(0..3, 0..1);

        drop(double_rpass);

        // Submit the first pass using both buffers
        ctx.queue.submit(Some(encoder1.finish()));
        // Drop the second buffer, meaning it's invalid to use draw
        // unless it's unbound.
        drop(vertex_buffer2);

        // Make sure the buffers are actually deleted.
        ctx.device.poll(Maintain::Wait);

        let mut encoder2 = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut single_rpass = encoder2.begin_render_pass(&RenderPassDescriptor {
            label: Some("single renderpass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: false,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: &[],
        });

        single_rpass.set_pipeline(&single_pipeline);
        single_rpass.set_vertex_buffer(0, vertex_buffer1.slice(..));
        single_rpass.draw(0..3, 0..1);

        drop(single_rpass);

        ctx.queue.submit(Some(encoder2.finish()));
    })
}
