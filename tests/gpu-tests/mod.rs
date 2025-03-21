automod::dir!("gpu-tests/");

mod regression {
    automod::dir!("gpu-tests/regression/");
}

wgpu_test::gpu_test_main!();
