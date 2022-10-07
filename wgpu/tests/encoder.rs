use crate::common::{initialize_test, TestParameters};

#[test]
fn drop_encoder() {
    initialize_test(TestParameters::default(), |ctx| {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder);
    })
}
