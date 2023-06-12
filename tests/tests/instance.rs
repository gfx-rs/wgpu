use wgpu_test::{infra::GpuTest, TestingContext};
#[derive(Clone, Default)]
pub struct InitializeTest;

impl GpuTest for InitializeTest {
    fn run(&self, _ctx: TestingContext) {}
}
