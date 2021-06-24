use crate::common::{initialize_test, TestParameters};

#[test]
fn device_initialization() {
    initialize_test(TestParameters::default(), |_ctx| {
        // intentionally empty
    })
}
