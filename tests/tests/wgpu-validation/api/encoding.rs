//! Tests of [`wgpu::CommandEncoder`] and related.

#[test]
fn as_hal() {
    // Sanity-test that the raw encoding API isn't completely broken.

    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.finish();
}

#[test]
#[should_panic = "Mixing the wgpu encoding API with the raw encoding API is not permitted"]
fn mix_apis_wgpu_then_hal() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&buffer, 0, None);
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
}

#[test]
#[should_panic = "Mixing the wgpu encoding API with the raw encoding API is not permitted"]
fn mix_apis_hal_then_wgpu() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.clear_buffer(&buffer, 0, None);
}

/// Test that the command encoder’s label is remembered and used in errors.
#[test]
#[should_panic = "In a CommandEncoder, label = 'my encoder'"]
fn encoding_error_contains_label_of_encoder() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("my buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("my encoder"),
    });
    // This is erroneous because it is copying to the same buffer.
    encoder.copy_buffer_to_buffer(&buffer, 0, &buffer, 0, 10);
    queue.submit([encoder.finish()]);
}
