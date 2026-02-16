use wgpu::{
    ColorTargetState, ColorWrites, Features, FragmentState, Limits, MeshPipelineDescriptor,
    MeshState, RenderPipelineDescriptor, ShaderModuleDescriptor, TextureFormat, VertexState,
};
use wgpu_test::{
    gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(PRIMITIVE_INDEX);
    tests.push(PRIMITIVE_INDEX_MESH_FRAGMENT);
    tests.push(PRIMITIVE_INDEX_MESH_NOT_FRAGMENT);
    tests.push(PRIMITIVE_INDEX_FRAGMENT_NOT_MESH);
}

async fn primitive_index(ctx: TestingContext) {
    const CODE: &str = "\
enable primitive_index;
@vertex
fn vertex() -> @builtin(position) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
@fragment
fn fragment(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(index), 1.0, 1.0, 1.0);
}
";
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CODE)),
    });
    let _ = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
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
            multiview_mask: None,
            cache: None,
        });
}

#[gpu_test]
static PRIMITIVE_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(Features::SHADER_PRIMITIVE_INDEX))
    .run_async(primitive_index);

async fn mesh_primitive_index(
    ctx: TestingContext,
    mesh_primitive_index: bool,
    fragment_primitive_index: bool,
) {
    const CODE: &str = "\
enable primitive_index;
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,MESH_PRIMITIVE_INDEX
}
struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitives) primitives: array<PrimitiveOutput, 3>,
    @builtin(primitive_count) primitive_count: u32,
}
var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@workgroup_size(1)
fn mesh() {
    mesh_output.vertex_count = 0;
    mesh_output.primitive_count = 0;
}
@fragment
fn fragment(FRAGMENT_PRIMITIVE_INDEX) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(index), 1.0, 1.0, 1.0);
}
";
    let used_code = CODE
        .replace(
            "MESH_PRIMITIVE_INDEX",
            if mesh_primitive_index {
                "\n@builtin(primitive_index) index: u32,"
            } else {
                ""
            },
        )
        .replace(
            "FRAGMENT_PRIMITIVE_INDEX",
            if fragment_primitive_index {
                "@builtin(primitive_index) index: u32"
            } else {
                ""
            },
        );
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&used_code)),
    });
    let _ = ctx.device.create_mesh_pipeline(&MeshPipelineDescriptor {
        label: None,
        layout: None,
        task: None,
        mesh: MeshState {
            module: &module,
            entry_point: None,
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
    // TODO: when naga support for MSL/HLSL mesh shaders lands enable Metal and DX12
    TestParameters::default()
        .features(Features::SHADER_PRIMITIVE_INDEX | Features::EXPERIMENTAL_MESH_SHADER)
        .limits(Limits::defaults().using_recommended_minimum_mesh_shader_values())
        .skip(FailureCase::backend(
            wgpu::Backends::METAL | wgpu::Backends::DX12,
        ))
}

#[gpu_test]
static PRIMITIVE_INDEX_MESH_FRAGMENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params())
    .run_async(async |ctx| mesh_primitive_index(ctx, true, true).await);

#[gpu_test]
static PRIMITIVE_INDEX_MESH_NOT_FRAGMENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params().expect_fail(FailureCase::always()))
    .run_async(async |ctx| mesh_primitive_index(ctx, true, false).await);

#[gpu_test]
static PRIMITIVE_INDEX_FRAGMENT_NOT_MESH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(mesh_params().expect_fail(FailureCase::always()))
    .run_async(async |ctx| mesh_primitive_index(ctx, false, true).await);
