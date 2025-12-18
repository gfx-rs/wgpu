// Same as in mesh shader tests
fn compile_wgsl(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    })
}
fn compile_hlsl(device: &wgpu::Device, entry: &str, stage_str: &str) -> wgpu::ShaderModule {
    let out_path = format!(
        "{}/src/mesh_shader/shader.{stage_str}.cso",
        env!("CARGO_MANIFEST_DIR")
    );
    let cmd = std::process::Command::new("dxc")
        .args([
            "-T",
            &format!("{stage_str}_6_5"),
            "-E",
            entry,
            &format!("{}/src/mesh_shader/shader.hlsl", env!("CARGO_MANIFEST_DIR")),
            "-Fo",
            &out_path,
        ])
        .output()
        .unwrap();
    if !cmd.status.success() {
        panic!("DXC failed:\n{}", String::from_utf8(cmd.stderr).unwrap());
    }
    let file = std::fs::read(&out_path).unwrap();
    std::fs::remove_file(out_path).unwrap();
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            entry_point: entry.to_owned(),
            label: None,
            num_workgroups: (1, 1, 1),
            dxil: Some(std::borrow::Cow::Owned(file)),
            ..Default::default()
        })
    }
}

fn compile_msl(device: &wgpu::Device, entry: &str) -> wgpu::ShaderModule {
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            entry_point: entry.to_owned(),
            label: None,
            msl: Some(std::borrow::Cow::Borrowed(include_str!("shader.metal"))),
            num_workgroups: (1, 1, 1),
            ..Default::default()
        })
    }
}

pub struct Example {
    pipeline: wgpu::RenderPipeline,
}
impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let (ts, ms, fs, ts_name, ms_name, fs_name) = match adapter.get_info().backend {
            wgpu::Backend::Vulkan => (
                compile_wgsl(device),
                compile_wgsl(device),
                compile_wgsl(device),
                "ts_main",
                "ms_main",
                "fs_main",
            ),
            wgpu::Backend::Dx12 => (
                compile_hlsl(device, "Task", "as"),
                compile_hlsl(device, "Mesh", "ms"),
                compile_hlsl(device, "Frag", "ps"),
                "main",
                "main",
                "main",
            ),
            wgpu::Backend::Metal => (
                compile_msl(device, "taskShader"),
                compile_msl(device, "meshShader"),
                compile_msl(device, "fragShader"),
                "main",
                "main",
                "main",
            ),
            _ => panic!("Example can currently only run on vulkan, dx12 or metal"),
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            task: Some(wgpu::TaskState {
                module: &ts,
                entry_point: Some(ts_name),
                compilation_options: Default::default(),
            }),
            mesh: wgpu::MeshState {
                module: &ms,
                entry_point: Some(ms_name),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some(fs_name),
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
                multiview_mask: None,
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

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "mesh_shader",
    image_path: "/examples/features/src/mesh_shader/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .features(
            wgpu::Features::EXPERIMENTAL_MESH_SHADER
                | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
        )
        .instance_flags(wgpu::InstanceFlags::advanced_debugging())
        .limits(wgpu::Limits::defaults().using_recommended_minimum_mesh_shader_values())
        .skip(wgpu_test::FailureCase {
            backends: None,
            // Skip Mesa because LLVMPIPE has what is believed to be a driver bug
            vendor: Some(0x10005),
            adapter: None,
            driver: None,
            reasons: vec![],
            behavior: wgpu_test::FailureBehavior::Ignore,
        }),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.005)],
    _phantom: std::marker::PhantomData::<Example>,
};
