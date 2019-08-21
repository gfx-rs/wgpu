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

#[derive(Clone, Copy)]
struct Vertex {
    _pos:   [f32; 2],
    _color: [f32; 4],
}

struct Example {
    vs_module: wgpu::ShaderModule,
    fs_module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
    multisampled_framebuffer: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    rebuild_pipeline: bool,
    sample_count: u32,
    sc_desc: wgpu::SwapChainDescriptor,
}

impl Example {
    fn create_pipeline(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, vs_module: &wgpu::ShaderModule, fs_module: &wgpu::ShaderModule, pipeline_layout: &wgpu::PipelineLayout, sample_count: u32) -> wgpu::RenderPipeline {
        println!("sample_count: {}", sample_count);
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::LineList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float4,
                        offset: 2 * 4,
                        shader_location: 1,
                    },
                ],
            }],
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        })
    }

    fn create_multisampled_framebuffer(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, sample_count: u32) -> wgpu::TextureView {
        let multisampled_texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };
        let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
            size: multisampled_texture_extent,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        };

        device.create_texture(multisampled_frame_descriptor).create_default_view()
    }
}

impl framework::Example for Example {
    fn init(sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) -> Self {
        println!("Press left/right arrow keys to change sample_count.");
        let sample_count = 4;

        let vs_bytes = framework::load_glsl(include_str!("shader.vert"), framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl(include_str!("shader.frag"), framework::ShaderStage::Fragment);
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[],
        });

        let pipeline = Example::create_pipeline(device, &sc_desc, &vs_module, &fs_module, &pipeline_layout, sample_count);
        let multisampled_framebuffer = Example::create_multisampled_framebuffer(device, sc_desc, sample_count);

        let mut vertex_data = vec!();

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

        let vertex_buffer = device
            .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(&vertex_data);
        let vertex_count = vertex_data.len() as u32;

        Example {
            vs_module,
            fs_module,
            pipeline_layout,
            pipeline,
            multisampled_framebuffer,
            vertex_buffer,
            vertex_count,
            rebuild_pipeline: false,
            sample_count,
            sc_desc: sc_desc.clone(),
        }
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { input, .. } => {
                if let winit::event::ElementState::Pressed = input.state {
                    match input.virtual_keycode {
                        Some(winit::event::VirtualKeyCode::Left) => {
                            if self.sample_count >= 2 {
                                self.sample_count = self.sample_count >> 1;
                                self.rebuild_pipeline = true;
                            }
                        }
                        Some(winit::event::VirtualKeyCode::Right) => {
                            if self.sample_count <= 16 {
                                self.sample_count = self.sample_count << 1;
                                self.rebuild_pipeline = true;
                            }
                        }
                        _ => { }
                    }
                }
            }
            _ => { }
        }
    }

    fn resize(&mut self, sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) {
        self.sc_desc = sc_desc.clone();
        self.multisampled_framebuffer = Example::create_multisampled_framebuffer(device, sc_desc, self.sample_count);
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        if self.rebuild_pipeline {
            self.pipeline = Example::create_pipeline(device, &self.sc_desc, &self.vs_module, &self.fs_module, &self.pipeline_layout, self.sample_count);
            self.multisampled_framebuffer = Example::create_multisampled_framebuffer(device, &self.sc_desc, self.sample_count);
            self.rebuild_pipeline = false;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        {
            let rpass_color_attachment = if self.sample_count == 1 {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                }
            } else {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &self.multisampled_framebuffer,
                    resolve_target: Some(&frame.view),
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                }
            };

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[rpass_color_attachment],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffers(0, &[(&self.vertex_buffer, 0)]);
            rpass.draw(0..self.vertex_count, 0..1);
        }

        device.get_queue().submit(&[encoder.finish()]);
    }
}

fn main() {
    framework::run::<Example>("msaa-line");
}
