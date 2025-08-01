use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(BUFFER_SIZE_AND_USAGE);
}

#[gpu_test]
static BUFFER_SIZE_AND_USAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
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
    });
