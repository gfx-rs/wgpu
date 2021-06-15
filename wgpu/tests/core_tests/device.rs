use wgpu::test::{TestParameters};

use super::initialize_test;

#[test]
fn device_initialization() {
    initialize_test(TestParameters::default(), |_ctx| {
        // intentionally empty
    })
}
