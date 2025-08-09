mod utils;

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    utils::all_tests(tests);
}
