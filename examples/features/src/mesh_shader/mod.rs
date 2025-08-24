use wgpu::include_wgsl;

pub struct Example {
    pipeline: wgpu::RenderPipeline,
}
impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let shader_module = unsafe {
            device.create_shader_module_trusted(
                include_wgsl!("shader.wgsl"),
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            task: Some(wgpu::TaskState {
                module: &shader_module,
                entry_point: Some("ts_main"),
                compilation_options: Default::default(),
            }),
            mesh: wgpu::MeshState {
                module: &shader_module,
                entry_point: Some("ms_main"),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self { pipeline }
    }
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.push_debug_group("Prepare data for draw.");
            rpass.set_pipeline(&self.pipeline);
            rpass.pop_debug_group();
            rpass.insert_debug_marker("Draw!");
            rpass.draw_mesh_tasks(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        Default::default()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_MESH_SHADER | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::defaults().using_recommended_minimum_mesh_shader_values()
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
    crate::framework::run::<Example>("mesh_shader");
}
