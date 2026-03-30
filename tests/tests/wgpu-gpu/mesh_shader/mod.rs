use wgpu::util::DeviceExt;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        MESH_PIPELINE_BASIC_MESH,
        MESH_PIPELINE_BASIC_TASK_MESH,
        MESH_PIPELINE_BASIC_MESH_FRAG,
        MESH_PIPELINE_BASIC_TASK_MESH_FRAG,
        MESH_DRAW,
        MESH_DRAW_NO_TASK,
        MESH_DRAW_DIVERGENT,
        MESH_DRAW_INDIRECT,
        MESH_MULTI_DRAW_INDIRECT,
        MESH_MULTI_DRAW_INDIRECT_COUNT,
        MESH_PIPELINE_BASIC_MESH_NO_DRAW,
        MESH_PIPELINE_BASIC_TASK_MESH_FRAG_NO_DRAW,
    ]);
}

// Same as in mesh shader example
fn compile_wgsl(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    })
}

struct Shaders {
    ts: Option<wgpu::ShaderModule>,
    ms: wgpu::ShaderModule,
    fs: Option<wgpu::ShaderModule>,
    ts_name: &'static str,
    ms_name: &'static str,
    fs_name: &'static str,
}
fn get_shaders(device: &wgpu::Device, info: &MeshPipelineTestInfo) -> Shaders {
    if info.divergent && info.use_task {
        unreachable!();
    }
    let compiled = compile_wgsl(device);
    Shaders {
        ts: info.use_task.then_some(compiled.clone()),
        ms: compiled.clone(),
        fs: info.use_frag.then_some(compiled),
        ts_name: "ts_main",
        ms_name: if info.divergent {
            "ms_divergent"
        } else if info.use_task {
            "ms_main"
        } else {
            "ms_no_ts"
        },
        fs_name: "fs_main",
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
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less), // 1.
        stencil: wgpu::StencilState::default(),           // 2.
        bias: wgpu::DepthBiasState::default(),
    };
    (depth_texture, depth_view, state)
}

struct MeshPipelineTestInfo {
    use_task: bool,
    use_frag: bool,
    draw: bool,
    divergent: bool,
}

fn mesh_pipeline_build(ctx: &TestingContext, info: MeshPipelineTestInfo) {
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);

    let Shaders {
        ts,
        ms,
        fs,
        ts_name,
        ms_name,
        fs_name,
    } = get_shaders(device, &info);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        task: ts.as_ref().map(|task| wgpu::TaskState {
            module: task,
            entry_point: Some(ts_name),
            compilation_options: Default::default(),
        }),
        mesh: wgpu::MeshState {
            module: &ms,
            entry_point: Some(ms_name),
            compilation_options: Default::default(),
        },
        fragment: fs.as_ref().map(|frag| wgpu::FragmentState {
            module: frag,
            entry_point: Some(fs_name),
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

fn mesh_draw(ctx: &TestingContext, draw_type: DrawType, info: MeshPipelineTestInfo) {
    let device = &ctx.device;
    let (_depth_image, depth_view, depth_state) = create_depth(device);

    let Shaders {
        ts,
        ms,
        fs,
        ts_name,
        ms_name,
        fs_name,
    } = get_shaders(device, &info);
    let frag = fs.unwrap();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        task: ts.as_ref().map(|task| wgpu::TaskState {
            module: task,
            entry_point: Some(ts_name),
            compilation_options: Default::default(),
        }),
        mesh: wgpu::MeshState {
            module: &ms,
            entry_point: Some(ms_name),
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag,
            entry_point: Some(fs_name),
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
            .features(
                wgpu::Features::EXPERIMENTAL_MESH_SHADER
                    | wgpu::Features::PASSTHROUGH_SHADERS
                    | match draw_type {
                        DrawType::Standard | DrawType::Indirect | DrawType::MultiIndirect => {
                            wgpu::Features::empty()
                        }
                        DrawType::MultiIndirectCount => wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
                    },
            )
            .limits(wgpu::Limits::default().using_recommended_minimum_mesh_shader_values())
            .skip(wgpu_test::FailureCase {
                backends: None,
                // Skip Mesa because LLVMPIPE has what is believed to be a driver bug
                vendor: Some(0x10005),
                adapter: None,
                driver: None,
                reasons: vec![],
                behavior: wgpu_test::FailureBehavior::Ignore,
            }),
    )
}

#[gpu_test]
pub static MESH_PIPELINE_BASIC_MESH: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_pipeline_build(
            &ctx,
            MeshPipelineTestInfo {
                use_task: false,
                use_frag: false,
                draw: true,
                divergent: false,
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
                use_frag: false,
                draw: true,
                divergent: false,
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
                use_frag: true,
                draw: true,
                divergent: false,
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
                use_frag: true,
                draw: true,
                divergent: false,
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
                use_frag: false,
                draw: false,
                divergent: false,
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
                use_frag: true,
                draw: false,
                divergent: false,
            },
        );
    });

// Mesh draw
#[gpu_test]
pub static MESH_DRAW: GpuTestConfiguration =
    default_gpu_test_config(DrawType::Standard).run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::Standard,
            MeshPipelineTestInfo {
                use_task: true,
                use_frag: true,
                draw: true,
                divergent: false,
            },
        );
    });
#[gpu_test]
pub static MESH_DRAW_NO_TASK: GpuTestConfiguration = default_gpu_test_config(DrawType::Standard)
    .run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::Standard,
            MeshPipelineTestInfo {
                use_task: false,
                use_frag: true,
                draw: true,
                divergent: false,
            },
        );
    });
#[gpu_test]
pub static MESH_DRAW_DIVERGENT: GpuTestConfiguration = default_gpu_test_config(DrawType::Standard)
    .run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::Standard,
            MeshPipelineTestInfo {
                use_task: false,
                use_frag: true,
                draw: true,
                divergent: true,
            },
        );
    });
#[gpu_test]
pub static MESH_DRAW_INDIRECT: GpuTestConfiguration = default_gpu_test_config(DrawType::Indirect)
    .run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::Indirect,
            MeshPipelineTestInfo {
                use_task: true,
                use_frag: true,
                draw: true,
                divergent: false,
            },
        );
    });
#[gpu_test]
pub static MESH_MULTI_DRAW_INDIRECT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirect).run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::MultiIndirect,
            MeshPipelineTestInfo {
                use_task: true,
                use_frag: true,
                draw: true,
                divergent: false,
            },
        );
    });
#[gpu_test]
pub static MESH_MULTI_DRAW_INDIRECT_COUNT: GpuTestConfiguration =
    default_gpu_test_config(DrawType::MultiIndirectCount).run_sync(|ctx| {
        mesh_draw(
            &ctx,
            DrawType::MultiIndirectCount,
            MeshPipelineTestInfo {
                use_task: true,
                use_frag: true,
                draw: true,
                divergent: false,
            },
        );
    });
