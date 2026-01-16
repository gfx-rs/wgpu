use wgpu::{
    ColorTargetState, ColorWrites, Features, FragmentState, RenderPipelineDescriptor,
    ShaderModuleDescriptor, TextureFormat, VertexState,
};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

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

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(DRAW_INDEX);
}

async fn test(ctx: TestingContext) {
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
    .parameters(TestParameters::default().features(Features::SHADER_DRAW_INDEX))
    .run_async(test);
