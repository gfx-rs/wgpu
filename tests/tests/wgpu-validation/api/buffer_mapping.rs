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

/// Regression test for [#9959]: `Buffer::unmap` racing a `Buffer::map` in
/// progress on another thread must never fail with `NotMapped`.
///
/// The main thread repeatedly requests a mapping and resolves it with `poll`;
/// the unmapper thread issues exactly one `unmap` per iteration, only after
/// that iteration's `map_async` has returned, with a swept spin delay so
/// successive iterations probe different points of the map. Once `map_async`
/// has been issued, the buffer is never observably unmapped until an `unmap`
/// succeeds: the racing `unmap` may abort the pending mapping or unmap the
/// freshly installed one, but it must not fail. Before the fix, `Buffer::map`
/// swapped `map_state` out for the duration of the HAL map, and an `unmap`
/// landing in that window failed with `NotMapped` even though its `map_async`
/// went on to resolve `Ok` — an interleaving no sequential ordering of the two
/// calls permits.
///
/// At the time of writing, unmodified trunk failed this test on every run,
/// with ~700 spurious failures per 25 000 iterations and the first one inside
/// the first dozen iterations. The full test takes ~0.25 s to 2 s, depending on machine.
/// Failure rate is machine-dependent too, so lowering the iterations risks false negatives.
///
/// [#9959]: https://github.com/gfx-rs/wgpu/pull/9959
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn concurrent_mapping_and_unmapping() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const ITERATIONS: u64 = 25_000;
    /// Spin-delay sweep period, in multiples of 4 `spin_loop` hints, covering
    /// the duration of the HAL map.
    const MAX_OFFSET: u64 = 250;

    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map/unmap race"),
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // `Buffer::unmap` reports failure through the uncaptured-error handler; a
    // hit is a change of this counter across the unmapper's call.
    let errors = Arc::new(AtomicUsize::new(0));
    let last_error = Arc::new(Mutex::new(String::new()));
    device.on_uncaptured_error(Arc::new({
        let errors = Arc::clone(&errors);
        let last_error = Arc::clone(&last_error);
        move |error: wgpu::Error| {
            *last_error.lock().unwrap() = error.to_string();
            errors.fetch_add(1, Ordering::SeqCst);
        }
    }));

    // Lockstep protocol: `map_issued = i + 1` publishes that iteration i's
    // `map_async` has returned; `unmap_done = i + 1` publishes that the
    // unmapper's verdict for iteration i is in `unmap_failed`.
    let map_issued = Arc::new(AtomicU64::new(0));
    let unmap_done = Arc::new(AtomicU64::new(0));
    let unmap_failed = Arc::new(AtomicU64::new(0));

    let unmapper = std::thread::spawn({
        let buffer = buffer.clone();
        let errors = Arc::clone(&errors);
        let map_issued = Arc::clone(&map_issued);
        let unmap_done = Arc::clone(&unmap_done);
        let unmap_failed = Arc::clone(&unmap_failed);
        move || {
            for i in 0..ITERATIONS {
                while map_issued.load(Ordering::SeqCst) != i + 1 {
                    std::hint::spin_loop();
                }
                for _ in 0..(i % MAX_OFFSET) * 4 {
                    std::hint::spin_loop();
                }
                let before = errors.load(Ordering::SeqCst);
                buffer.unmap();
                let failed = errors.load(Ordering::SeqCst) != before;
                unmap_failed.store(failed as u64, Ordering::SeqCst);
                unmap_done.store(i + 1, Ordering::SeqCst);
            }
        }
    });

    let map_result = Arc::new(Mutex::new(None::<Result<(), wgpu::BufferAsyncError>>));
    let mut hits = Vec::new();
    for i in 0..ITERATIONS {
        map_result.lock().unwrap().take();
        buffer.map_async(wgpu::MapMode::Read, .., {
            let map_result = Arc::clone(&map_result);
            move |result| {
                *map_result.lock().unwrap() = Some(result);
            }
        });
        map_issued.store(i + 1, Ordering::SeqCst);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        while unmap_done.load(Ordering::SeqCst) != i + 1 {
            std::hint::spin_loop();
        }

        if unmap_failed.load(Ordering::SeqCst) != 0 {
            let map_ok = matches!(*map_result.lock().unwrap(), Some(Ok(())));
            hits.push(format!(
                "iteration {i}: unmap failed ({}) while its map_async {}",
                last_error.lock().unwrap(),
                if map_ok {
                    "resolved Ok"
                } else {
                    "did not resolve Ok"
                },
            ));
            if map_ok {
                // The spurious failure left the mapping in place; really unmap
                // so the next iteration starts idle.
                buffer.unmap();
            }
        }
    }
    unmapper.join().unwrap();

    assert!(
        hits.is_empty(),
        "unmap must never fail while racing a map_async it was issued after \
         ({} spurious failures in {ITERATIONS} iterations; first: {})",
        hits.len(),
        hits[0],
    );
}
