mod core_tests {
    // All files containing tests
    mod device;
    mod instance;
    mod vertex_indexes;

    fn initialize_test(
        parameters: wgpu::test::TestParameters,
        test_function: impl FnOnce(&mut wgpu::test::TestingContext<'_>),
    ) {
        // We don't actually care if it fails
        let _ = env_logger::try_init();

        pollster::block_on(wgpu::test::initialize_test_core(parameters, test_function));
    }
}
