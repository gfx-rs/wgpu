//! Tests of [`wgpu::Buffer`] and related.

/// Ensures that submitting a command buffer referencing an already destroyed buffer
/// results in an error.
#[test]
#[should_panic = "Buffer with '' label has been destroyed"]
fn destroyed_buffer() {
    let (device, queues) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let queue = queues.into_iter().next().unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        initial_queue: None,
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        queue: None,
        label: None,
    });
    encoder.clear_buffer(&buffer, 0, None);

    buffer.destroy();

    queue.submit([encoder.finish()]);
}
