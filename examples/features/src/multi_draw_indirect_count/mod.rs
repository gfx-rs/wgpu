use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const EXAMPLE_NAME: &str = "multi_draw_indirect_count";

const NUM_QUADS: u32 = 16;
const QUAD_SIZE: f32 = 0.2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

struct Example {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x3,
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let cols = 4;
        let mut vertices = Vec::new();
        let mut indirect_args = Vec::new();

        for i in 0..NUM_QUADS {
            let row = i / cols;
            let col = i % cols;
            let cx = -0.75 + col as f32 * 0.5;
            let cy = 0.75 - row as f32 * 0.5;
            let s = QUAD_SIZE;

            let hue = i as f32 / NUM_QUADS as f32;
            let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);

            vertices.extend_from_slice(&[
                Vertex {
                    position: [cx - s, cy - s],
                    color: [r, g, b],
                },
                Vertex {
                    position: [cx + s, cy - s],
                    color: [r, g, b],
                },
                Vertex {
                    position: [cx - s, cy + s],
                    color: [r, g, b],
                },
                Vertex {
                    position: [cx + s, cy - s],
                    color: [r, g, b],
                },
                Vertex {
                    position: [cx + s, cy + s],
                    color: [r, g, b],
                },
                Vertex {
                    position: [cx - s, cy + s],
                    color: [r, g, b],
                },
            ]);

            let first_vertex = i * 6;
            indirect_args.push(wgpu::util::DrawIndirectArgs {
                vertex_count: 6,
                instance_count: 1,
                first_vertex,
                first_instance: 0,
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indirect Buffer"),
            contents: bytemuck::cast_slice(&indirect_args),
            usage: wgpu::BufferUsages::INDIRECT,
        });

        let count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Count Buffer"),
            contents: bytemuck::cast_slice::<u32, u8>(&[NUM_QUADS]),
            usage: wgpu::BufferUsages::INDIRECT,
        });

        Example {
            pipeline,
            vertex_buffer,
            indirect_buffer,
            count_buffer,
        }
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.multi_draw_indirect_count(
                &self.indirect_buffer,
                0,
                &self.count_buffer,
                0,
                NUM_QUADS,
            );
        }

        queue.submit([encoder.finish()]);
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0) as i32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

pub fn main() {
    crate::framework::run::<Example>(EXAMPLE_NAME);
}

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: EXAMPLE_NAME,
    image_path: "/examples/features/src/multi_draw_indirect_count/screenshot.png",
    width: 256,
    height: 256,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default(),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
