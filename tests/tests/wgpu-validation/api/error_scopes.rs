#![cfg(not(target_arch = "wasm32"))]
use std::{
    panic::{resume_unwind, AssertUnwindSafe},
    sync::Arc,
};

use parking_lot::Mutex;

const ERR: &str = "Buffer size 9223372036854775808 is greater than the maximum buffer size";
fn raise_validation_error(device: &wgpu::Device) {
    let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1 << 63, // Too large!
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
}

fn register_uncaptured_error_handler(device: &wgpu::Device) -> Arc<Mutex<Vec<wgpu::Error>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let errors_clone = errors.clone();

    device.on_uncaptured_error(Arc::new(move |error| {
        errors_clone.lock().push(error);
    }));

    errors
}

fn assert_matches_string(error: &wgpu::Error, substr: &str) {
    let err_str = error.to_string();
    assert!(
        err_str.contains(substr),
        "Error string '{err_str}' does not contain expected substring '{substr}'"
    );
}

// Test that error scopes work correctly in the basic case.
#[test]
fn basic() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    raise_validation_error(&device);
    let error = pollster::block_on(scope.pop());

    assert!(error.is_some());
}

// Test that error scopes are thread-local: an error scope pushed on one thread
// does not capture errors generated on another thread.
#[test]
fn multi_threaded_scopes() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    // Start an error scope on the main thread.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Register an uncaptured error handler to catch errors from other threads.
    let other_thread_error = register_uncaptured_error_handler(&device);

    // Do something invalid on another thread.
    std::thread::scope(|s| {
        s.spawn(|| {
            raise_validation_error(&device);
        });
    });

    // Pop the error scope on the main thread.
    let error = pollster::block_on(scope.pop());

    // The main thread's error scope should not have captured the other thread's error.
    assert!(error.is_none());
    // The other thread's error should have been reported to the uncaptured error handler.
    let uncaptured_errors = other_thread_error.lock();
    assert_eq!(uncaptured_errors.len(), 1);
    assert_matches_string(&uncaptured_errors[0], ERR);
}

// Test that error scopes error when popped in the wrong order.
#[test]
#[should_panic(expected = "error scopes must be popped in reverse order")]
fn pop_out_of_order() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let scope1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _scope2 = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let _ = pollster::block_on(scope1.pop());
}

// Test that error scopes are automatically popped when dropped.
#[test]
fn drop_automatically_pops() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let uncaptured_error = register_uncaptured_error_handler(&device);

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    raise_validation_error(&device);
    drop(scope); // Automatically pops the error scope.

    assert!(uncaptured_error.lock().is_empty());

    // Raising another error will go to the uncaptured error handler, not the dropped scope.
    raise_validation_error(&device);

    assert_eq!(uncaptured_error.lock().len(), 1);
    assert_matches_string(&uncaptured_error.lock()[0], ERR);
}

// Test that error scopes are automatically popped when dropped during unwinding,
// even when they are dropped out of order.
#[test]
fn drop_during_unwind() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let scope1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let scope2 = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
        raise_validation_error(&device);
        // Move scope1 so that it is dropped before scope2.
        let _scope2 = scope2;
        let _scope1 = scope1;
        resume_unwind(Box::new("unwind"))
    }));

    assert_eq!(*res.unwrap_err().downcast_ref::<&str>().unwrap(), "unwind");
}
