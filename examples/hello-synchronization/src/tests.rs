use super::*;
use pollster::FutureExt;
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
            let ExecuteResults {
                patient_workgroup_results,
                hasty_workgroup_results: _,
            } = execute(
                &ctx.device,
                &ctx.queue,
                ARR_SIZE
            ).block_on();
            assert_eq!(patient_workgroup_results, [16_u32; ARR_SIZE]);
        },
    );
}
