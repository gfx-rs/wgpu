use crate::common::{initialize_test, TestParameters};
use wasm_bindgen_test::*;

#[test]
#[wasm_bindgen_test]
fn drop_encoder() {
    initialize_test(TestParameters::default(), |ctx| {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder);
    })
}
