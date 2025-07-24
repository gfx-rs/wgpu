use std::{io::Write, process::Stdio};

use wgpu::util::DeviceExt;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

// Same as in mesh shader example
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
    println!("{shader_stage}");
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

fn mesh_pipeline_build(
    ctx: &TestingContext,
    task: Option<&[u8]>,
    mesh: &[u8],
    frag: Option<&[u8]>,
    draw: bool,
) {
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);
    let task = task.map(|t| compile_glsl(device, t, "task"));
    let mesh = compile_glsl(device, mesh, "mesh");
    let frag = frag.map(|f| compile_glsl(device, f, "frag"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
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
    if draw {
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
            });
            pass.set_pipeline(&pipeline);
            pass.draw_mesh_tasks(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::PollType::Wait).unwrap();
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
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);
    let task = compile_glsl(device, BASIC_TASK, "task");
    let mesh = compile_glsl(device, BASIC_MESH, "mesh");
    let frag = compile_glsl(device, NO_WRITE_FRAG, "frag");
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
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
    ctx.device.poll(wgpu::PollType::Wait).unwrap();
}

const BASIC_TASK: &[u8] = include_bytes!("basic.task");
const BASIC_MESH: &[u8] = include_bytes!("basic.mesh");
//const BASIC_FRAG: &[u8] = include_bytes!("basic.frag.spv");
const NO_WRITE_FRAG: &[u8] = include_bytes!("no-write.frag");

fn default_gpu_test_config(draw_type: DrawType) -> GpuTestConfiguration {
    GpuTestConfiguration::new().parameters(
        TestParameters::default()
            .test_features_limits()
            .features(
                wgpu::Features::EXPERIMENTAL_MESH_SHADER
                    | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                    | match draw_type {
                        DrawType::Standard | DrawType::Indirect => wgpu::Features::empty(),
                        DrawType::MultiIndirect => wgpu::Features::MULTI_DRAW_INDIRECT,
                        DrawType::MultiIndirectCount => wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
                    },
            )
            .limits(wgpu::Limits::default().using_recommended_minimum_mesh_shader_values()),
    )
}

// Mesh pipeline configs
#[gpu_test]
static MESH_PIPELINE_BASIC_MESH: GpuTestConfiguration = default_gpu_test_config(DrawType::Standard)
    .run_sync(|ctx| {
        mesh_pipeline_build(&ctx, None, BASIC_MESH, None, true);
    });
#[gpu_test]
static MESH_PIPELINE_BASIC_TASK_MESH: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(&ctx, Some(BASIC_TASK), BASIC_MESH, None, true);
    });
#[gpu_test]
static MESH_PIPELINE_BASIC_MESH_FRAG: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(&ctx, None, BASIC_MESH, Some(NO_WRITE_FRAG), true);
    });
#[gpu_test]
static MESH_PIPELINE_BASIC_TASK_MESH_FRAG: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            Some(BASIC_TASK),
            BASIC_MESH,
            Some(NO_WRITE_FRAG),
            true,
        );
    });

// Mesh draw
#[gpu_test]
static MESH_DRAW_INDIRECT: GpuTestConfiguration = default_gpu_test_config(DrawType::Indirect)
    .run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::Indirect);
    });
#[gpu_test]
static MESH_MULTI_DRAW_INDIRECT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirect).run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::MultiIndirect);
    });
#[gpu_test]
static MESH_MULTI_DRAW_INDIRECT_COUNT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirectCount).run_sync(|ctx| {
        mesh_draw(&ctx, DrawType::MultiIndirectCount);
    });
