//! Shader module labels are handed to the backend's shader compiler, which may
//! interpret them as more than an opaque string. DXC in particular reads the
//! label as the input file's path, so a label containing a `:` used to fail to
//! compile at all.

use wgpu::{DownlevelFlags, Limits};
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(SHADER_MODULE_LABEL_PUNCTUATION);
}

const SHADER: &str = r#"
    @compute @workgroup_size(1)
    fn main() {}
"#;

/// Labels containing punctuation that a shader compiler might read as path
/// syntax must not affect compilation.
#[apply(gpu_test!)]
static SHADER_MODULE_LABEL_PUNCTUATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        for label in [
            "Renderer::shader",
            "Renderer:shader",
            "a::b::c",
            "renderer/shader",
            "C:/shader",
            "shader with spaces",
            "shader\twith\ttabs",
            "/foo",
            "-foo",
            "",
        ] {
            create_pipeline(&ctx, label);
        }
    });

fn create_pipeline(ctx: &TestingContext, label: &str) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });

    // The backend compiles the shader here, not in `create_shader_module`.
    let _pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
}
