use std::{
    hash::{DefaultHasher, Hash, Hasher},
    process::Stdio,
};

use wgpu::util::DeviceExt;
use wgpu_test::{
    fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        MESH_PIPELINE_BASIC_MESH,
        MESH_PIPELINE_BASIC_TASK_MESH,
        MESH_PIPELINE_BASIC_MESH_FRAG,
        MESH_PIPELINE_BASIC_TASK_MESH_FRAG,
        MESH_DRAW_INDIRECT,
        MESH_MULTI_DRAW_INDIRECT,
        MESH_MULTI_DRAW_INDIRECT_COUNT,
        MESH_PIPELINE_BASIC_MESH_NO_DRAW,
        MESH_PIPELINE_BASIC_TASK_MESH_FRAG_NO_DRAW,
        MESH_DISABLED,
    ]);
}

// Same as in mesh shader example
fn compile_glsl(device: &wgpu::Device, shader_stage: &'static str) -> wgpu::ShaderModule {
    let cmd = std::process::Command::new("glslc")
        .args([
            &format!(
                "{}/tests/wgpu-gpu/mesh_shader/basic.{shader_stage}",
                env!("CARGO_MANIFEST_DIR")
            ),
            "-o",
            "-",
            "--target-env=vulkan1.2",
            "--target-spv=spv1.4",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to call glslc");
    let output = cmd.wait_with_output().expect("Error waiting for glslc");
    assert!(output.status.success());
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            entry_point: "main".into(),
            label: None,
            spirv: Some(wgpu::util::make_spirv_raw(&output.stdout)),
            ..Default::default()
        })
    }
}

fn compile_hlsl(
    device: &wgpu::Device,
    entry: &str,
    stage_str: &str,
    test_name: &str,
) -> wgpu::ShaderModule {
    // Each test needs its own files
    let out_path = format!(
        "{}/tests/wgpu-gpu/mesh_shader/{test_name}.{stage_str}.cso",
        env!("CARGO_MANIFEST_DIR")
    );
    let cmd = std::process::Command::new("dxc")
        .args([
            "-T",
            &format!("{stage_str}_6_5"),
            "-E",
            entry,
            &format!(
                "{}/tests/wgpu-gpu/mesh_shader/basic.hlsl",
                env!("CARGO_MANIFEST_DIR")
            ),
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

fn get_shaders(
    device: &wgpu::Device,
    backend: wgpu::Backend,
    test_name: &str,
    info: &MeshPipelineTestInfo,
) -> (
    Option<wgpu::ShaderModule>,
    wgpu::ShaderModule,
    Option<wgpu::ShaderModule>,
) {
    // On backends that don't support mesh shaders, or for the MESH_DISABLED
    // test, compile a dummy shader so we can construct a structurally valid
    // pipeline description and test that `create_mesh_pipeline` fails.
    // (In the case that the platform does support mesh shaders, the dummy
    // shader is used to avoid requiring EXPERIMENTAL_PASSTHROUGH_SHADERS.)
    let dummy_shader = device.create_shader_module(wgpu::include_wgsl!("non_mesh.wgsl"));
    match backend {
        wgpu::Backend::Vulkan => (
            info.use_task.then(|| compile_glsl(device, "task")),
            if info.use_mesh {
                compile_glsl(device, "mesh")
            } else {
                dummy_shader
            },
            info.use_frag.then(|| compile_glsl(device, "frag")),
        ),
        wgpu::Backend::Dx12 => (
            info.use_task
                .then(|| compile_hlsl(device, "Task", "as", test_name)),
            if info.use_mesh {
                compile_hlsl(device, "Mesh", "ms", test_name)
            } else {
                dummy_shader
            },
            info.use_frag
                .then(|| compile_hlsl(device, "Frag", "ps", test_name)),
        ),
        wgpu::Backend::Metal => (
            info.use_task.then(|| compile_msl(device, "taskShader")),
            if info.use_mesh {
                compile_msl(device, "meshShader")
            } else {
                dummy_shader
            },
            info.use_frag.then(|| compile_msl(device, "fragShader")),
        ),
        _ => {
            assert!(!info.use_task && !info.use_mesh && !info.use_frag);
            (None, dummy_shader, None)
        }
    }
}

fn create_depth(
    device: &wgpu::Device,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::DepthStencilState) {
    let image_size = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: image_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&Default::default());
    let state = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less, // 1.
        stencil: wgpu::StencilState::default(),     // 2.
        bias: wgpu::DepthBiasState::default(),
    };
    (depth_texture, depth_view, state)
}

struct MeshPipelineTestInfo {
    use_task: bool,
    use_mesh: bool,
    use_frag: bool,
    draw: bool,
}

fn hash_testing_context(ctx: &TestingContext) -> u64 {
    let mut hasher = DefaultHasher::new();
    ctx.hash(&mut hasher);
    hasher.finish()
}

fn mesh_pipeline_build(ctx: &TestingContext, info: MeshPipelineTestInfo) {
    let backend = ctx.adapter.get_info().backend;
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);

    let test_hash = hash_testing_context(ctx).to_string();
    let (task, mesh, frag) = get_shaders(device, backend, &test_hash, &info);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        task: task.as_ref().map(|task| wgpu::TaskState {
            module: task,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        }),
        mesh: wgpu::MeshState {
            module: &mesh,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        },
        fragment: frag.as_ref().map(|frag| wgpu::FragmentState {
            module: frag,
            entry_point: Some("main"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(depth_state),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    if info.draw {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw_mesh_tasks(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum DrawType {
    #[allow(dead_code)]
    Standard,
    Indirect,
    MultiIndirect,
    MultiIndirectCount,
}

fn mesh_draw(ctx: &TestingContext, draw_type: DrawType) {
    let backend = ctx.adapter.get_info().backend;
    if backend != wgpu::Backend::Vulkan && backend != wgpu::Backend::Dx12 {
        return;
    }
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);
    let test_hash = hash_testing_context(ctx).to_string();
    let info = MeshPipelineTestInfo {
        use_task: true,
        use_mesh: true,
        use_frag: true,
        draw: true,
    };
    let (task, mesh, frag) = get_shaders(device, backend, &test_hash, &info);
    let task = task.unwrap();
    let frag = frag.unwrap();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        task: Some(wgpu::TaskState {
            module: &task,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        }),
        mesh: wgpu::MeshState {
            module: &mesh,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag,
            entry_point: Some("main"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(depth_state),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    let buffer = match draw_type {
        DrawType::Standard => None,
        DrawType::Indirect | DrawType::MultiIndirect | DrawType::MultiIndirectCount => Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                usage: wgpu::BufferUsages::INDIRECT,
                contents: bytemuck::bytes_of(&[1u32; 4]),
            }),
        ),
    };
    let count_buffer = match draw_type {
        DrawType::MultiIndirectCount => Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                usage: wgpu::BufferUsages::INDIRECT,
                contents: bytemuck::bytes_of(&[1u32; 1]),
            },
        )),
        _ => None,
    };
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        match draw_type {
            DrawType::Standard => pass.draw_mesh_tasks(1, 1, 1),
            DrawType::Indirect => pass.draw_mesh_tasks_indirect(buffer.as_ref().unwrap(), 0),
            DrawType::MultiIndirect => {
                pass.multi_draw_mesh_tasks_indirect(buffer.as_ref().unwrap(), 0, 1)
            }
            DrawType::MultiIndirectCount => pass.multi_draw_mesh_tasks_indirect_count(
                buffer.as_ref().unwrap(),
                0,
                count_buffer.as_ref().unwrap(),
                0,
                1,
            ),
        }
        pass.draw_mesh_tasks_indirect(buffer.as_ref().unwrap(), 0);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

fn default_gpu_test_config(draw_type: DrawType) -> GpuTestConfiguration {
    GpuTestConfiguration::new().parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::GPU_BASED_VALIDATION)
            .test_features_limits()
            .features(
                wgpu::Features::EXPERIMENTAL_MESH_SHADER
                    | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS
                    | match draw_type {
                        DrawType::Standard | DrawType::Indirect | DrawType::MultiIndirect => {
                            wgpu::Features::empty()
                        }
                        DrawType::MultiIndirectCount => wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
                    },
            )
            .limits(wgpu::Limits::default().using_recommended_minimum_mesh_shader_values()),
    )
}

#[gpu_test]
pub static MESH_PIPELINE_BASIC_MESH: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: false,
                use_mesh: true,
                use_frag: false,
                draw: true,
            },
        );
    });
#[gpu_test]
pub static MESH_PIPELINE_BASIC_TASK_MESH: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: true,
                use_mesh: true,
                use_frag: false,
                draw: true,
            },
        );
    });
#[gpu_test]
pub static MESH_PIPELINE_BASIC_MESH_FRAG: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: false,
                use_mesh: true,
                use_frag: true,
                draw: true,
            },
        );
    });
#[gpu_test]
pub static MESH_PIPELINE_BASIC_TASK_MESH_FRAG: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: true,
                use_mesh: true,
                use_frag: true,
                draw: true,
            },
        );
    });
#[gpu_test]
pub static MESH_PIPELINE_BASIC_MESH_NO_DRAW: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: false,
                use_mesh: true,
                use_frag: false,
                draw: false,
            },
        );
    });
#[gpu_test]
pub static MESH_PIPELINE_BASIC_TASK_MESH_FRAG_NO_DRAW: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: true,
                use_mesh: true,
                use_frag: true,
                draw: false,
            },
        );
    });

// Mesh draw
#[gpu_test]
pub static MESH_DRAW_INDIRECT: GpuTestConfiguration = default_gpu_test_config(DrawType::Indirect)
    .run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::Indirect);
    });
#[gpu_test]
pub static MESH_MULTI_DRAW_INDIRECT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirect).run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::MultiIndirect);
    });
#[gpu_test]
pub static MESH_MULTI_DRAW_INDIRECT_COUNT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirectCount).run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::MultiIndirectCount);
    });

/// When the mesh shading feature is disabled, calls to `create_mesh_pipeline`
/// should be rejected. This should be the case on all backends, not just the
/// ones where the feature could be turned on.
#[gpu_test]
pub static MESH_DISABLED: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    fail(
        &ctx.device,
        || {
            mesh_pipeline_build(
                &ctx,
                MeshPipelineTestInfo {
                    use_task: false,
                    use_mesh: false,
                    use_frag: false,
                    draw: true,
                },
            );
        },
        Some(concat![
            "Features Features { ",
            "features_wgpu: FeaturesWGPU(EXPERIMENTAL_MESH_SHADER), ",
            "features_webgpu: FeaturesWebGPU(0x0) ",
            "} are required but not enabled on the device",
        ]),
    )
});
