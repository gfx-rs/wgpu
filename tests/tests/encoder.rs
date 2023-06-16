use wgpu_test::{infra::GpuTest, TestingContext};

#[derive(Default)]
pub struct DropEncoderTest;

impl GpuTest for DropEncoderTest {
    fn run(&self, ctx: TestingContext) {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder);
    }
}
