use wasm_bindgen_test::*;
use wgpu_test::{initialize_test, TestParameters};

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
