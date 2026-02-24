use wgpu::{Features, Limits};
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext, gpu_test};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        PIPELINE_CREATE_USE
    ]);
}

#[gpu_test]
static PIPELINE_CREATE_USE: GpuTestConfiguration = GpuTestConfiguration::new().parameters(TestParameters::default().features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES).limits(Limits::defaults().using_minimum_supported_acceleration_structure_values())).run_sync(pipeline_create_use);

fn pipeline_create_use(ctx: TestingContext) {
    let ray_gen_source = "@group(0) @binding(0) acc_struct: acceleration_structure;
    
    ";
}