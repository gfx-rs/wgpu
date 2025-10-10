use std::{num::NonZero, time::Instant};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt, TextureBlitter},
    vertex_attr_array,
};

const TEXTURE_SIZE: u32 = 512;

pub struct Example {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    view: wgpu::TextureView,
    view1: wgpu::TextureView,
    view2: wgpu::TextureView,
    start_time: Instant,
    blitter: TextureBlitter,
}
impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let shader_src = "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
            }

            @fragment
            fn fs_main(@builtin(view_index) view_index: u32) -> @location(0) vec4f {
                return vec4f(f32(view_index) * 0.25 + 0.125);
            }
        ";

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            vertex: wgpu::VertexState {
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                }],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: NonZero::new(2),
            multisample: Default::default(),
            layout: None,
            depth_stencil: None,
            cache: None,
        });
        let vertex_buffer_content: &[f32; 12] = &[
            // Triangle 1
            -1.0, -1.0, // Bottom left
            1.0, 1.0, // Top right
            -1.0, 1.0, // Top left
            // Triangle 2
            -1.0, -1.0, // Bottom left
            1.0, -1.0, // Bottom right
            1.0, 1.0, // Top right
        ];
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vertex_buffer_content),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::R8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(2),
        });
        let view1 = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::R8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let view2 = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::R8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 1,
            array_layer_count: Some(1),
        });
        let blitter = wgpu::util::TextureBlitter::new(device, config.format);
        Self {
            pipeline,
            vertex_buffer,
            view,
            view1,
            view2,
            blitter,
            start_time: Instant::now(),
        }
    }
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        unsafe {
            device.start_graphics_debugger_capture();
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: NonZero::new(3),
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
        if (Instant::now() - self.start_time)
            .as_secs()
            .is_multiple_of(2)
        {
            self.blitter.copy(device, &mut encoder, &self.view1, view);
        } else {
            self.blitter.copy(device, &mut encoder, &self.view2, view);
        }
        queue.submit(Some(encoder.finish()));
        unsafe {
            device.stop_graphics_debugger_capture();
        }
    }
    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        Default::default()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::MULTIVIEW
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_multiview_view_count: 2,
            ..Default::default()
        }
    }
    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        // empty
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {
        // empty
    }
}

pub fn main() {
    crate::framework::run::<Example>("multiview");
}
