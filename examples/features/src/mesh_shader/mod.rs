// Same as in mesh shader tests
fn compile_wgsl(device: &wgpu::Device) -> wgpu::ShaderModule {
    // Workgroup memory zero initialization can be expensive for mesh shaders
    unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    }
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
            label: None,
            num_workgroups: (1, 1, 1),
            dxil: Some(std::borrow::Cow::Owned(file)),
            ..Default::default()
        })
    }
}

fn compile_msl(device: &wgpu::Device) -> wgpu::ShaderModule {
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: None,
            msl: Some(std::borrow::Cow::Borrowed(include_str!("shader.metal"))),
            num_workgroups: (1, 1, 1),
            ..Default::default()
        })
    }
}

struct Shaders {
    ts: wgpu::ShaderModule,
    ms: wgpu::ShaderModule,
    fs: wgpu::ShaderModule,
    ts_name: &'static str,
    ms_name: &'static str,
    fs_name: &'static str,
}

fn get_shaders(device: &wgpu::Device, backend: wgpu::Backend) -> Shaders {
    // In the case that the platform does support mesh shaders, the dummy
    // shader is used to avoid requiring PASSTHROUGH_SHADERS.
    match backend {
        wgpu::Backend::Vulkan => {
            let compiled = compile_wgsl(device);
            Shaders {
                ts: compiled.clone(),
                ms: compiled.clone(),
                fs: compiled.clone(),
                ts_name: "ts_main",
                ms_name: "ms_main",
                fs_name: "fs_main",
            }
        }
        wgpu::Backend::Dx12 => Shaders {
            ts: compile_hlsl(device, "Task", "as"),
            ms: compile_hlsl(device, "Mesh", "ms"),
            fs: compile_hlsl(device, "Frag", "ps"),
            ts_name: "main",
            ms_name: "main",
            fs_name: "main",
        },
        wgpu::Backend::Metal => {
            let compiled = compile_msl(device);
            Shaders {
                ts: compiled.clone(),
                ms: compiled.clone(),
                fs: compiled.clone(),
                ts_name: "taskShader",
                ms_name: "meshShader",
                fs_name: "fragShader",
            }
        }
        _ => unreachable!(),
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
        let Shaders {
            ts,
            ms,
            fs,
            ts_name,
            ms_name,
            fs_name,
        } = get_shaders(device, adapter.get_info().backend);
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
        wgpu::Features::EXPERIMENTAL_MESH_SHADER | wgpu::Features::PASSTHROUGH_SHADERS
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
        .features(wgpu::Features::EXPERIMENTAL_MESH_SHADER | wgpu::Features::PASSTHROUGH_SHADERS)
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
