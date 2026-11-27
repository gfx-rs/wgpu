//! Tests of [`wgpu::Buffer`] and related.

/// Ensures that submitting a command buffer referencing an already destroyed buffer
/// results in an error.
#[test]
#[should_panic = "Buffer with '' label has been destroyed"]
fn destroyed_buffer() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&buffer, 0, None);

    buffer.destroy();

    queue.submit([encoder.finish()]);
}

#[test]
fn acceleration_structure_inputs_require_features() {
    let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    for usage in [
        wgpu::BufferUsages::BLAS_INPUT,
        wgpu::BufferUsages::TLAS_INPUT,
        wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::TLAS_INPUT,
    ] {
        wgpu_test::fail(
            &device,
            || {
                let _ = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 36,
                    usage,
                    mapped_at_creation: false,
                });
            },
            Some("required but not enabled"),
        );
    }
}
