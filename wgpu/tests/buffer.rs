use crate::common::{initialize_test, TestParameters};

#[test]
fn buffer_size_and_usage_get() {
    initialize_test(TestParameters::default(), |ctx| {
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
    })
}
