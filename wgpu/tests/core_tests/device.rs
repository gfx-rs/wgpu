use crate::core_tests::common::init::{initialize_test, TestParameters};

#[test]
fn device_initialization() {
    initialize_test(TestParameters::default(), |_ctx| {
        // intentionally empty
    })
}
