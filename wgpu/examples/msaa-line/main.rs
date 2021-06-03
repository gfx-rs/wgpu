//! The parts of this example enabling MSAA are:
//! *    The render pipeline is created with a sample_count > 1.
//! *    A new texture with a sample_count > 1 is created and set as the color_attachment instead of the swapchain.
//! *    The swapchain is now specified as a resolve_target.
//!
//! The parts of this example enabling LineList are:
//! *   Set the primitive_topology to PrimitiveTopology::LineList.
//! *   Vertices and Indices describe the two points that make up a line.

#[path = "../framework.rs"]
mod framework;

use std::{borrow::Cow, iter};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _color: [f32; 4],
}

struct Example {
    bundle: wgpu::RenderBundle,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    multisampled_framebuffer: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    sample_count: u32,
    rebuild_bundle: bool,
    sc_desc: wgpu::SwapChainDescriptor,
}

impl Example {
    fn create_bundle(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        sample_count: u32,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: u32,
    ) -> wgpu::RenderBundle {
        log::info!("sample_count: {}", sample_count);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[sc_desc.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
        });
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[sc_desc.format],
                depth_stencil_format: None,
                sample_count,
            });
        encoder.set_pipeline(&pipeline);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.draw(0..vertex_count, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("main"),
        })
    }

    fn create_multisampled_framebuffer(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let multisampled_texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth_or_array_layers: 1,
        };
        let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
            size: multisampled_texture_extent,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            label: None,
        };

        device
            .create_texture(multisampled_frame_descriptor)
            .create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl framework::Example for Example {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        log::info!("Press left/right arrow keys to change sample_count.");
        let sample_count = 4;

        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Metal | wgpu::Backend::Vulkan => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION
            }
            _ => (), //TODO
        }
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            flags,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let multisampled_framebuffer =
            Example::create_multisampled_framebuffer(device, sc_desc, sample_count);

        let mut vertex_data = vec![];

        let max = 50;
        for i in 0..max {
            let percent = i as f32 / max as f32;
            let (sin, cos) = (percent * 2.0 * std::f32::consts::PI).sin_cos();
            vertex_data.push(Vertex {
                _pos: [0.0, 0.0],
                _color: [1.0, -sin, cos, 1.0],
            });
            vertex_data.push(Vertex {
                _pos: [1.0 * cos, 1.0 * sin],
                _color: [sin, -cos, 1.0, 1.0],
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsage::VERTEX,
        });
        let vertex_count = vertex_data.len() as u32;

        let bundle = Example::create_bundle(
            device,
            &sc_desc,
            &shader,
            &pipeline_layout,
            sample_count,
            &vertex_buffer,
            vertex_count,
        );

        Example {
            bundle,
            shader,
            pipeline_layout,
            multisampled_framebuffer,
            vertex_buffer,
            vertex_count,
            sample_count,
            rebuild_bundle: false,
            sc_desc: sc_desc.clone(),
        }
    }

    #[allow(clippy::single_match)]
    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { input, .. } => {
                if let winit::event::ElementState::Pressed = input.state {
                    match input.virtual_keycode {
                        // TODO: Switch back to full scans of possible options when we expose
                        //       supported sample counts to the user.
                        Some(winit::event::VirtualKeyCode::Left) => {
                            if self.sample_count == 4 {
                                self.sample_count = 1;
                                self.rebuild_bundle = true;
                            }
                        }
                        Some(winit::event::VirtualKeyCode::Right) => {
                            if self.sample_count == 1 {
                                self.sample_count = 4;
                                self.rebuild_bundle = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.sc_desc = sc_desc.clone();
        self.multisampled_framebuffer =
            Example::create_multisampled_framebuffer(device, sc_desc, self.sample_count);
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        if self.rebuild_bundle {
            self.bundle = Example::create_bundle(
                device,
                &self.sc_desc,
                &self.shader,
                &self.pipeline_layout,
                self.sample_count,
                &self.vertex_buffer,
                self.vertex_count,
            );
            self.multisampled_framebuffer =
                Example::create_multisampled_framebuffer(device, &self.sc_desc, self.sample_count);
            self.rebuild_bundle = false;
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            };
            let rpass_color_attachment = if self.sample_count == 1 {
                wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops,
                }
            } else {
                wgpu::RenderPassColorAttachment {
                    view: &self.multisampled_framebuffer,
                    resolve_target: Some(&frame.view),
                    ops,
                }
            };

            encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[rpass_color_attachment],
                    depth_stencil_attachment: None,
                })
                .execute_bundles(iter::once(&self.bundle));
        }

        queue.submit(iter::once(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("msaa-line");
}
