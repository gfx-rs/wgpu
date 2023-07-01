use super::*;
use wgpu_test::{initialize_test, TestParameters};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn hello_synchronization_test_results() {
    initialize_test(
        // Taken from hello-compute tests.
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .specific_failure(None, None, Some("V3D"), true),
        |ctx| {
            let mut local_patient_results = [0_u32; ARR_SIZE];
            let mut local_hasty_results = [0_u32; ARR_SIZE];
            pollster::block_on(execute(
                &ctx.device,
                &ctx.queue,
                &mut local_patient_results[..],
                &mut local_hasty_results[..]
            ));
            assert_eq!(local_patient_results, [16_u32; ARR_SIZE]);
        }
    );
}