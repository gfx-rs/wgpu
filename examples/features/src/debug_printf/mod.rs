struct Example {}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::DEBUG_PRINTF
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }

    fn init(
        _config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let shader_source = r#"
       enable wgpu_debug_printf;

        @compute @workgroup_size(8, 1, 1)
        fn main(@builtin(local_invocation_index) idx: u32) {
            // We use a specific ID to identify our print in the logs
             debugPrintf("WGSL_METAL_LOG: Thread index is %u", idx);
        }
    "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DebugPrintfShader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create a simple pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Debug Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(Some(encoder.finish()));

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        Example {}
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    fn render(&mut self, _view: &wgpu::TextureView, _device: &wgpu::Device, _queue: &wgpu::Queue) {}
}

pub fn main() {
    crate::framework::run::<Example>("debug-printf");
}
