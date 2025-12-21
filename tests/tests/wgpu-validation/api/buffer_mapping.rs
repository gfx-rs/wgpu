fn mapping_is_zeroed(array: &[u8]) {
    for (i, &byte) in array.iter().enumerate() {
        assert_eq!(byte, 0, "Byte at index {i} is not zero");
    }
}

// Ensure that a simple immutable mapping works and it is zeroed.
#[test]
fn full_immutable_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    buffer.map_async(wgpu::MapMode::Read, .., |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mapping = buffer.slice(..).get_mapped_range();

    mapping_is_zeroed(&mapping);

    drop(mapping);

    buffer.unmap();
}

// Ensure that a simple mutable binding works and it is zeroed.
#[test]
fn full_mut_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let mapping = buffer.slice(..).get_mapped_range_mut();

    mapping_is_zeroed(&mapping);

    drop(mapping);

    buffer.unmap();
}

// Ensure that you can make two non-overlapping immutable ranges, which are both zeroed
#[test]
fn split_immutable_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    buffer.map_async(wgpu::MapMode::Read, .., |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mapping0 = buffer.slice(0..512).get_mapped_range();
    let mapping1 = buffer.slice(512..1024).get_mapped_range();

    mapping_is_zeroed(&mapping0);
    mapping_is_zeroed(&mapping1);

    drop(mapping0);
    drop(mapping1);

    buffer.unmap();
}

/// Ensure that you can make two non-overlapping mapped ranges, which are both zeroed
#[test]
fn split_mut_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let mapping0 = buffer.slice(0..512).get_mapped_range_mut();
    let mapping1 = buffer.slice(512..1024).get_mapped_range_mut();

    mapping_is_zeroed(&mapping0);
    mapping_is_zeroed(&mapping1);

    drop(mapping0);
    drop(mapping1);

    buffer.unmap();
}

/// Ensure that you can make two overlapping immutablely mapped ranges.
#[test]
fn overlapping_ref_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let _mapping0 = buffer.slice(0..512).get_mapped_range();
    let _mapping1 = buffer.slice(256..768).get_mapped_range();
}

/// Ensure that two overlapping mutably mapped ranges panics.
#[test]
#[should_panic(expected = "break Rust memory aliasing rules")]
fn overlapping_mut_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let _mapping0 = buffer.slice(0..512).get_mapped_range_mut();
    let _mapping1 = buffer.slice(256..768).get_mapped_range_mut();
}

/// Ensure that when you try to get a mapped range from an unmapped buffer, it panics with
/// an error mentioning a completely unmapped buffer.
#[test]
#[should_panic(expected = "an unmapped buffer")]
fn not_mapped() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let _mapping = buffer.slice(..).get_mapped_range_mut();
}

/// Ensure that when you partially map a buffer, then try to read outside of that range, it panics
/// mentioning the mapped indices.
#[test]
#[should_panic(
    expected = "Attempted to get range 512..1024 (Mutable), but the mapped range is 0..512"
)]
fn partially_mapped() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    buffer.map_async(wgpu::MapMode::Write, 0..512, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let _mapping0 = buffer.slice(0..512).get_mapped_range_mut();
    let _mapping1 = buffer.slice(512..1024).get_mapped_range_mut();
}

/// Ensure that you cannot unmap a buffer while there are still accessible mapped views.
#[test]
#[should_panic(expected = "You cannot unmap a buffer that still has accessible mapped views")]
fn unmap_while_visible() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let _mapping0 = buffer.slice(..).get_mapped_range_mut();
    buffer.unmap();
}
