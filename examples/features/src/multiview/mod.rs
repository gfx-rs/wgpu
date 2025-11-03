//! Renders different content to different layers of an array texture using multiview,
//! a feature commonly used for VR rendering.

use std::{num::NonZero, time::Instant};

use wgpu::util::TextureBlitter;

const TEXTURE_SIZE: u32 = 512;

// Change this to demonstrate non-contiguous multiview functionality
const LAYER_MASK: u32 = 0b11;

const NUM_LAYERS: u32 = 32 - LAYER_MASK.leading_zeros();

pub struct Example {
    pipeline: wgpu::RenderPipeline,
    entire_texture_view: wgpu::TextureView,
    views: Vec<wgpu::TextureView>,
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
        let shader_src = include_str!("shader.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            vertex: wgpu::VertexState {
                buffers: &[],
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: NonZero::new(LAYER_MASK),
            multisample: Default::default(),
            layout: None,
            depth_stencil: None,
            cache: None,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth_or_array_layers: NUM_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let entire_texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(NUM_LAYERS),
        });
        let mut views = Vec::new();
        for i in 0..NUM_LAYERS {
            views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                dimension: Some(wgpu::TextureViewDimension::D2),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: Some(1),
            }));
        }
        let blitter = wgpu::util::TextureBlitter::new(device, config.format);
        Self {
            pipeline,
            entire_texture_view,
            views,
            blitter,
            start_time: Instant::now(),
        }
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.entire_texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: NonZero::new(LAYER_MASK),
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.draw(0..6, 0..1);
        }

        let layer = (Instant::now() - self.start_time).as_secs() % NUM_LAYERS as u64;
        self.blitter
            .copy(device, &mut encoder, &self.views[layer as usize], view);
        queue.submit(Some(encoder.finish()));
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        Default::default()
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::MULTIVIEW
            | if !(LAYER_MASK + 1).is_power_of_two() {
                wgpu::Features::SELECTIVE_MULTIVIEW
            } else {
                wgpu::Features::empty()
            }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_multiview_view_count: NUM_LAYERS,
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
