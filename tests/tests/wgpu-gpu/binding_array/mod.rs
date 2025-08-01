mod buffers;
mod sampled_textures;
mod samplers;
mod storage_textures;

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    buffers::all_tests(tests);
    sampled_textures::all_tests(tests);
    samplers::all_tests(tests);
    storage_textures::all_tests(tests);
}
