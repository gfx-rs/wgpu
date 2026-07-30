use wgpu_test::{
    apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(LINEAR_INTERPOLATION_GATED);
}

const SHADER_SRC: &str = "
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(f32(vertex_index), 0.0, 0.0, 1.0);
    out.v = 1.0;
    return out;
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(linear) v: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.v);
}
";

/// `@interpolate(linear)` has no GLSL ES equivalent, so on adapters without
/// [`wgpu::DownlevelFlags::LINEAR_INTERPOLATION`] it must be rejected up front by
/// `create_shader_module`, instead of failing later during pipeline creation with an
/// internal naga bitflag name.
///
/// Regression test for <https://github.com/gfx-rs/wgpu/issues/9971>.
fn linear_interpolation_gated(ctx: TestingContext) {
    let supported = ctx
        .adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::LINEAR_INTERPOLATION);

    let create = || {
        ctx.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linear-interpolation"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            })
    };

    if supported {
        create();
    } else {
        fail(&ctx.device, create, Some("LINEAR_INTERPOLATION"));
    }
}

#[apply(gpu_test!)]
static LINEAR_INTERPOLATION_GATED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::empty())
            .limits(wgpu::Limits::downlevel_webgl2_defaults()),
    )
    .run_sync(linear_interpolation_gated);
