// FIXME: Now that MAP_WRITE mappings are write-only,
// the “mut” and “immutable” terminology is incorrect.

fn read_mapping_is_zeroed(slice: &[u8]) {
    for (i, &byte) in slice.iter().enumerate() {
        assert_eq!(byte, 0, "Byte at index {i} is not zero");
    }
}
fn write_mapping_is_zeroed(mut slice: wgpu::WriteOnly<'_, [u8]>) {
    let ptr = slice.as_raw_ptr().cast::<u8>();
    for i in 0..slice.len() {
        assert_eq!(
            // SAFETY: it is not, in general, safe to read from a write mapping, but our goal here
            // is specifically to verify the internally provided zeroedness.
            //
            // FIXME: Is the goal of these tests to ensure that zeroes are what is exposed to Rust,
            // and not to ensure that zeroes get into the GPU buffer? If so, then we can delete
            // them, or perhaps replace them with tests of mapping without writing, then reading.
            unsafe { ptr.add(i).read() },
            0,
            "Byte at index {i} is not zero"
        );
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

    let mapping = buffer.slice(..).get_mapped_range().unwrap();

    read_mapping_is_zeroed(&mapping);

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

    let mut mapping = buffer.slice(..).get_mapped_range_mut().unwrap();

    write_mapping_is_zeroed(mapping.slice(..));

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

    let mapping0 = buffer.slice(0..512).get_mapped_range().unwrap();
    let mapping1 = buffer.slice(512..1024).get_mapped_range().unwrap();

    read_mapping_is_zeroed(&mapping0);
    read_mapping_is_zeroed(&mapping1);

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

    let mut mapping0 = buffer.slice(0..512).get_mapped_range_mut().unwrap();
    let mut mapping1 = buffer.slice(512..1024).get_mapped_range_mut().unwrap();

    write_mapping_is_zeroed(mapping0.slice(..));
    write_mapping_is_zeroed(mapping1.slice(..));

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

    let _mapping0 = buffer.slice(0..512).get_mapped_range().unwrap();
    let _mapping1 = buffer.slice(256..768).get_mapped_range().unwrap();
}

/// Ensure that two overlapping mutably mapped ranges returns an error.
#[test]
fn overlapping_mut_binding() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });

    let _mapping0 = buffer.slice(0..512).get_mapped_range_mut().unwrap();
    let result = buffer.slice(256..768).get_mapped_range_mut();
    assert!(result.is_err());
}

/// Ensure that getting a mapped range from an unmapped buffer returns an error.
#[test]
fn not_mapped() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let result = buffer.slice(..).get_mapped_range_mut();
    assert!(result.is_err());
}

/// Ensure that getting a mapped range outside the mapped region returns an error.
#[test]
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

    let _mapping0 = buffer.slice(0..512).get_mapped_range_mut().unwrap();
    let result = buffer.slice(512..1024).get_mapped_range_mut();
    assert!(result.is_err());
}

/// Ensure that `map_async` calls its callback even when the buffer is invalid.
///
/// Regression test: when `buffer_map_async` failed because the buffer id referred to
/// an invalid buffer (one whose creation had failed), it dropped `op` via `?` without
/// calling its callback, violating the documented guarantee that the callback is always
/// called.
#[test]
fn map_async_on_invalid_buffer_calls_callback() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    // MAP_READ | MAP_WRITE is an invalid usage combination, so create_buffer
    // will fail and the returned buffer will be invalid. Capture the error so
    // the default (panic) handler is not reached.
    let _creation_error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("invalid"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });
    drop(_creation_error_scope);

    // `map_async` on an invalid buffer should fire the callback with an error.
    // Also capture the Err that wgpu-core returns to wgpu's map_async layer, which
    // wgpu forwards to the error sink regardless of whether it called the callback.
    let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let callback_called2 = callback_called.clone();
    let _map_error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        assert!(result.is_err(), "expected an error for an invalid buffer");
        callback_called2.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    drop(_map_error_scope);

    assert!(
        callback_called.load(std::sync::atomic::Ordering::SeqCst),
        "map_async callback was not called for an invalid buffer"
    );
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

    let _mapping0 = buffer.slice(..).get_mapped_range_mut().unwrap();
    buffer.unmap();
}
