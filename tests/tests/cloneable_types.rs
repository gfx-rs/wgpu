use wgpu_test::{gpu_test, TestingContext};

#[gpu_test]
static CLONEABLE_BUFFERS: GpuTestConfiguration =
    wgpu_test::GpuTestConfiguration::new().run_sync(cloneable_buffers);

// Test a basic case of cloneable types where you clone the buffer to be able
// to access the buffer inside the callback as well as outside.
fn cloneable_buffers(ctx: TestingContext) {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: true,
    });

    let buffer_contents: Vec<u8> = (0..32).collect();

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(&buffer_contents);

    buffer.unmap();

    // This is actually a bug, we should not need to call submit to make the buffer contents visible.
    ctx.queue.submit([]);

    let cloned_buffer = buffer.clone();
    let cloned_buffer_contents = buffer_contents.clone();

    buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
        let data = cloned_buffer.slice(..).get_mapped_range();

        assert_eq!(&*data, &cloned_buffer_contents);
    });

    ctx.device.poll(wgpu::Maintain::Wait);

    let data = buffer.slice(..).get_mapped_range();

    assert_eq!(&*data, &buffer_contents);
}
