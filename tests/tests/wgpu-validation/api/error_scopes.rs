#![cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// Test that error scopes are thread-local: an error scope pushed on one thread
// does not capture errors generated on another thread.
#[test]
fn multi_threaded_scopes() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let other_thread_error = Arc::new(AtomicBool::new(false));
    let other_thread_error_clone = other_thread_error.clone();

    // Start an error scope on the main thread.
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Register an uncaptured error handler to catch errors from other threads.
    device.on_uncaptured_error(Arc::new(move |_error| {
        other_thread_error_clone.store(true, Ordering::Relaxed);
    }));

    // Do something invalid on another thread.
    std::thread::scope(|s| {
        s.spawn(|| {
            let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1 << 63, // Too large!
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        });
    });

    // Pop the error scope on the main thread.
    let error = pollster::block_on(device.pop_error_scope());

    // The main thread's error scope should not have captured the other thread's error.
    assert!(error.is_none());
    // The other thread's error should have been reported to the uncaptured error handler.
    assert!(other_thread_error.load(Ordering::Relaxed));
}
