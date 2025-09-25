use crate::framework;

pub struct Example {
    pipeline: wgpu::RenderPipeline,
}
impl framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let vs_shader = unsafe {
            device.create_shader_module_passthrough(wgpu::include_precompiled_wgsl!(
                "src/hello_triangle/shader.wgsl",
                "vs_main",
                all
            ))
        };
        let fs_shader = unsafe {
            device.create_shader_module_passthrough(wgpu::include_precompiled_wgsl!(
                "src/hello_triangle/shader.wgsl",
                "fs_main",
                all
            ))
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
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
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {}
    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }
    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::defaults()
    }
}

pub fn main() {
    crate::framework::run::<Example>("precompiled_shader");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "precompiled_shader",
    image_path: "/examples/features/src/mesh_shader/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .test_features_limits()
        .features(wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
