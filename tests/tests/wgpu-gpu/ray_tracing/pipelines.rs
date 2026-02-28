use wgpu::{Features, Limits, ShaderModuleDescriptor};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(PIPELINE_CREATE_USE);
}

#[gpu_test]
static PIPELINE_CREATE_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(Limits::defaults().using_minimum_supported_acceleration_structure_values()),
    )
    .run_sync(pipeline_create_use);

fn pipeline_create_use(ctx: TestingContext) {
    let ray_gen_source = "@group(0) @binding(0) acc_struct: acceleration_structure;
        var<ray_payload> payload: u32;

        @ray_generation
        fn gen() {
            traceRays(acc_struct, RayDesc(), &payload);
        }

        @closest_hit
        fn closest() {
            
        }
    ";

    let ray_closest_source = "
        var<incoming_ray_payload> payload: u32;

        @incoming_payload(payload)
        @closest_hit
        fn closest() {
            
        }
    ";

    let ray_any_source = "
        var<incoming_ray_payload> payload: u32;

        @incoming_payload(payload)
        @any_hit
        fn any() {
            
        }
    ";

    let ray_miss_source = "
        var<incoming_ray_payload> payload: u32;

        @incoming_payload(payload)
        @any_hit
        fn miss() {
            
        }
    ";

    let ray_gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_gen_source)),
    });

    let ray_closest = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray closest hit shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_closest_source)),
    });

    let ray_any = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray any hit shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_any_source)),
    });

    let ray_miss = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray miss shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_miss_source)),
    });
}
