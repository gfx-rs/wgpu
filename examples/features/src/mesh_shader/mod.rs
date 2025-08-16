use std::{io::Write, process::Stdio};

// Same as in mesh shader tests
fn compile_glsl(
    device: &wgpu::Device,
    data: &[u8],
    shader_stage: &'static str,
) -> wgpu::ShaderModule {
    let cmd = std::process::Command::new("glslc")
        .args([
            &format!("-fshader-stage={shader_stage}"),
            "-",
            "-o",
            "-",
            "--target-env=vulkan1.2",
            "--target-spv=spv1.4",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to call glslc");
    cmd.stdin.as_ref().unwrap().write_all(data).unwrap();
    let output = cmd.wait_with_output().expect("Error waiting for glslc");
    assert!(output.status.success());
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough::SpirV(
            wgpu::ShaderModuleDescriptorSpirV {
                label: None,
                source: wgpu::util::make_spirv_raw(&output.stdout),
            },
        ))
    }
}
fn compile_hlsl(
    device: &wgpu::Device,
    data: &str,
    entry: &str,
    stage_str: &str,
) -> wgpu::ShaderModule {
    // I dont understand why but for some reason the amplification shader specifically can't be from dxc command line on my system?
    // Dx12 complains that it is corrupted, even though dxc -dumpbin succeeds and yields what seems to be an identical shader
    // to the one generated from naga's dxc code. I have also verified that the code passed to DXIL passthrough is identical
    // to what is passed to directx down the line. I am very very confused.
    // Also have verified that it isn't the DXC version by forcing my own dynamic DXC
    if stage_str != "as" {
        let out_path = format!(
            "{}/src/mesh_shader/.{stage_str}.dxil",
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
                "-HV",
                "2018",
            ])
            .output()
            .unwrap();
        if !cmd.status.success() {
            panic!("DXC failed:\n{}", String::from_utf8(cmd.stderr).unwrap());
        }
        let file = std::fs::read(&out_path).unwrap();
        std::fs::remove_file(out_path).unwrap();
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough::Dxil(
                wgpu::ShaderModuleDescriptorDxil {
                    entry_point: entry.to_owned(),
                    label: None,
                    source: &file,
                    num_workgroups: (0, 0, 0),
                },
            ))
        }
    } else {
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough::Hlsl(
                wgpu::ShaderModuleDescriptorHlsl {
                    entry_point: entry.to_owned(),
                    label: None,
                    source: data,
                    num_workgroups: (0, 0, 0),
                },
            ))
        }
    }
}

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
        let features = device.features();
        let (ts, ms, fs) = if features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
            (
                compile_glsl(device, include_bytes!("shader.task"), "task"),
                compile_glsl(device, include_bytes!("shader.mesh"), "mesh"),
                compile_glsl(device, include_bytes!("shader.frag"), "frag"),
            )
        } else if features.contains(wgpu::Features::HLSL_DXIL_SHADER_PASSTHROUGH) {
            (
                compile_hlsl(device, include_str!("shader.hlsl"), "Task", "as"),
                compile_hlsl(device, include_str!("shader.hlsl"), "Mesh", "ms"),
                compile_hlsl(device, include_str!("shader.hlsl"), "Frag", "ps"),
            )
        } else {
            panic!("Device must support SPIRV_SHADER_PASSTHROUGH or HLSL_DXIL_SHADER_PASSTHROUGH");
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            task: Some(wgpu::TaskState {
                module: &ts,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            }),
            mesh: wgpu::MeshState {
                module: &ms,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some("main"),
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
        wgpu::Features::EXPERIMENTAL_MESH_SHADER
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::defaults().using_recommended_minimum_mesh_shader_values()
    }
    fn optional_features() -> wgpu::Features {
        wgpu::Features::SPIRV_SHADER_PASSTHROUGH | wgpu::Features::HLSL_DXIL_SHADER_PASSTHROUGH
    }
    fn backend_options() -> Option<wgpu::BackendOptions> {
        Some(wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        })
    }
    fn supported_backends() -> wgpu::Backends {
        wgpu::Backends::VULKAN | wgpu::Backends::DX12
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
