use wasm_bindgen_test::*;

use wgpu_examples::test_common::{initialize_test, TestParameters};

#[test]
#[wasm_bindgen_test]
fn device_initialization() {
    initialize_test(TestParameters::default(), |_ctx| {
        // intentionally empty
    })
}
