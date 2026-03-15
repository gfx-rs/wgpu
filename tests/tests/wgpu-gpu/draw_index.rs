use wgpu::{
    Backends, ColorTargetState, ColorWrites, Features, FragmentState, Limits,
    MeshPipelineDescriptor, MeshState, RenderPipelineDescriptor, ShaderModuleDescriptor, TaskState,
    TextureFormat, VertexState,
};
use wgpu_test::{
    gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(DRAW_INDEX);
    tests.push(DRAW_INDEX_MESH_NO_TASK);
    tests.push(DRAW_INDEX_MESH_TASK);
    tests.push(DRAW_INDEX_TASK_NO_MESH);
}

async fn test(ctx: TestingContext) {
    const CODE: &str = "\
enable draw_index;

struct Input {
    @builtin(draw_index) draw_index: u32,
}

@vertex
fn vertex(input: Input) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(input.draw_index), 1.0, 1.0, 1.0);
}
@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
";
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CODE)),
    });
    let _pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: &module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
}

#[gpu_test]
static DRAW_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_DRAW_INDEX)
            // https://github.com/gfx-rs/wgpu/issues/9184
            .expect_fail(
                wgpu_test::FailureCase::molten_vk()
                    .validation_error("could not be compiled into pipeline")
                    .validation_error("vkDestroyDevice")
                    .panic("Unexpected Vulkan error: ERROR_INITIALIZATION_FAILED"),
            ),
    )
    .run_async(test);

async fn test_mesh(ctx: TestingContext, use_task: bool, mesh_uses_draw_id: bool) {
    const CODE: &str = "\
enable draw_index;
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}
struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitives) primitives: array<PrimitiveOutput, 3>,
    @builtin(primitive_count) primitive_count: u32,
}
var<workgroup> mesh_output: MeshOutput;
struct TaskPayload { value: u32 }
var<task_payload> payload: TaskPayload;

@task
@payload(payload)
@workgroup_size(1)
fn task(@builtin(draw_index) id: u32) -> @builtin(mesh_task_size) vec3<u32> {
    return vec3<u32>(1, 1, 1);
}

@mesh(mesh_output)
@payload(payload)
@workgroup_size(1)
fn mesh_ts(MESH_DRAW_INDEX1) {
    mesh_output.vertex_count = 0;
    mesh_output.primitive_count = 0;
}

@mesh(mesh_output)
@workgroup_size(1)
fn mesh_no_ts(MESH_DRAW_INDEX2) {
    mesh_output.vertex_count = 0;
    mesh_output.primitive_count = 0;
}
@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
";

    let used_code = CODE
        .replace(
            "MESH_DRAW_INDEX2",
            if mesh_uses_draw_id {
                "@builtin(draw_index) id: u32"
            } else {
                ""
            },
        )
        .replace(
            "MESH_DRAW_INDEX1",
            if mesh_uses_draw_id && use_task {
                "@builtin(draw_index) id: u32"
            } else {
                ""
            },
        );
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(used_code)),
    });
    let _ = ctx.device.create_mesh_pipeline(&MeshPipelineDescriptor {
        label: None,
        layout: None,
        task: use_task.then_some(TaskState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
        }),
        mesh: MeshState {
            module: &module,
            entry_point: Some(if use_task { "mesh_ts" } else { "mesh_no_ts" }),
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::all(),
            })],
        }),
        multiview: None,
        cache: None,
    });
}

fn mesh_params() -> TestParameters {
    // TODO: when support for mesh shaders in naga on dx12/metal lands enable those backends
    TestParameters::default()
        .features(Features::SHADER_DRAW_INDEX | Features::EXPERIMENTAL_MESH_SHADER)
        .skip(FailureCase::backend(Backends::DX12 | Backends::METAL))
        .limits(Limits::defaults().using_recommended_minimum_mesh_shader_values())
}

#[gpu_test]
static DRAW_INDEX_TASK_NO_MESH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params())
    .run_async(async |ctx| test_mesh(ctx, true, false).await);

#[gpu_test]
static DRAW_INDEX_MESH_NO_TASK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params())
    .run_async(async |ctx| test_mesh(ctx, false, true).await);

#[gpu_test]
static DRAW_INDEX_MESH_TASK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params().expect_fail(FailureCase::always()))
    .run_async(async |ctx| test_mesh(ctx, true, true).await);
