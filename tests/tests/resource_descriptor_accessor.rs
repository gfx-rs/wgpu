use wgpu_test::infra::GpuTest;

/// Buffer's size and usage can be read back.
#[derive(Default)]
pub struct BufferSizeAndUsageTest;

impl GpuTest for BufferSizeAndUsageTest {
    fn run(&self, ctx: wgpu_test::TestingContext) {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1234,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        assert_eq!(buffer.size(), 1234);
        assert_eq!(
            buffer.usage(),
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
        );
    }
}
