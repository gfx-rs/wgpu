use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::Arc;

/// Helper to create a small mappable buffer for READ tests.
fn make_read_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// map_buffer_on_submit defers mapping until submit, then invokes the callback after polling.
#[test]
fn encoder_map_buffer_on_submit_defers_until_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 16);

    let fired = Arc::new(AtomicBool::new(false));
    let fired_cl = Arc::clone(&fired);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });

    // Register deferred map.
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, 0..4, move |_| {
        fired_cl.store(true, SeqCst);
    });
    // Include a trivial command that uses the buffer.
    encoder.clear_buffer(&buffer, 0, None);

    // Polling before submit should not trigger the callback.
    _ = device.poll(wgpu::PollType::Poll);
    assert!(!fired.load(SeqCst));

    // Submit and wait; callback should fire.
    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());
    assert!(fired.load(SeqCst));
}

/// Out-of-bounds ranges panic during submit (when the deferred map executes).
#[test]
#[should_panic = "is out of range for buffer of size"]
fn encoder_map_buffer_on_submit_out_of_bounds_panics_on_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 16);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    // 12..24 overflows the 16-byte buffer (size=12, end=24).
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, 12..24, |_| {});
    encoder.clear_buffer(&buffer, 0, None);

    // Panic happens inside submit when executing deferred actions.
    queue.submit([encoder.finish()]);
}

/// If the buffer is already mapped when the deferred mapping executes, it panics during submit.
#[test]
#[should_panic = "Buffer with 'read buffer' label is still mapped"]
fn encoder_map_buffer_on_submit_panics_if_already_mapped_on_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 16);

    // Start a mapping now so the buffer is considered mapped.
    buffer.slice(0..4).map_async(wgpu::MapMode::Read, |_| {});

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    // Deferred mapping of an already-mapped buffer will panic when executed on submit or be rejected by submit.
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, 0..4, |_| {});
    // Include any trivial work; using the same buffer ensures core validation catches the mapped hazard.
    encoder.clear_buffer(&buffer, 0, None);

    queue.submit([encoder.finish()]);
}

/// on_submitted_work_done is deferred until submit.
#[test]
fn encoder_on_submitted_work_done_defers_until_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let fired = Arc::new(AtomicBool::new(false));
    let fired_cl = Arc::clone(&fired);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.on_submitted_work_done(move || {
        fired_cl.store(true, SeqCst);
    });

    // Include a trivial command so the command buffer isn't completely empty.
    let dummy = make_read_buffer(&device, 4);
    encoder.clear_buffer(&dummy, 0, None);

    // Without submission, polling shouldn't invoke the callback.
    _ = device.poll(wgpu::PollType::Poll);
    assert!(!fired.load(SeqCst));

    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());
    assert!(fired.load(SeqCst));
}

/// Both kinds of deferred callbacks are enqueued and eventually invoked.
#[test]
fn encoder_both_callbacks_fire_after_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 16);

    let map_fired = Arc::new(AtomicBool::new(false));
    let map_fired_cl = Arc::clone(&map_fired);
    let queue_fired = Arc::new(AtomicBool::new(false));
    let queue_fired_cl = Arc::clone(&queue_fired);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, 0..4, move |_| {
        map_fired_cl.store(true, SeqCst);
    });
    encoder.on_submitted_work_done(move || {
        queue_fired_cl.store(true, SeqCst);
    });
    encoder.clear_buffer(&buffer, 0, None);

    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());

    assert!(map_fired.load(SeqCst));
    assert!(queue_fired.load(SeqCst));
}

/// Registering multiple deferred mappings works; all callbacks fire after submit.
#[test]
fn encoder_multiple_map_buffer_on_submit_callbacks_fire() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer1 = make_read_buffer(&device, 32);
    let buffer2 = make_read_buffer(&device, 32);

    let counter = Arc::new(AtomicU32::new(0));
    let c1 = Arc::clone(&counter);
    let c2 = Arc::clone(&counter);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.map_buffer_on_submit(&buffer1, wgpu::MapMode::Read, 0..4, move |_| {
        c1.fetch_add(1, SeqCst);
    });
    encoder.map_buffer_on_submit(&buffer2, wgpu::MapMode::Read, 8..12, move |_| {
        c2.fetch_add(1, SeqCst);
    });
    encoder.clear_buffer(&buffer1, 0, None);

    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(counter.load(SeqCst), 2);
}

/// Mapping with a buffer lacking MAP_* usage should panic when executed on submit.
#[test]
#[should_panic]
fn encoder_map_buffer_on_submit_panics_if_usage_invalid_on_submit() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let unmappable = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unmappable buffer"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST, // No MAP_READ or MAP_WRITE
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.map_buffer_on_submit(&unmappable, wgpu::MapMode::Read, 0..4, |_| {});

    // Add unrelated work so the submission isn't empty.
    let dummy = make_read_buffer(&device, 4);
    encoder.clear_buffer(&dummy, 0, None);

    // Panic expected when deferred mapping executes.
    queue.submit([encoder.finish()]);
}

/// Deferred map callbacks run before on_submitted_work_done for the same submission.
#[test]
fn encoder_deferred_map_runs_before_on_submitted_work_done() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 16);

    #[derive(Default)]
    struct Order {
        map_order: AtomicU32,
        queue_order: AtomicU32,
        counter: AtomicU32,
    }
    let order = Arc::new(Order::default());
    let o_map = Arc::clone(&order);
    let o_queue = Arc::clone(&order);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, 0..4, move |_| {
        let v = o_map.counter.fetch_add(1, SeqCst);
        o_map.map_order.store(v, SeqCst);
    });
    encoder.on_submitted_work_done(move || {
        let v = o_queue.counter.fetch_add(1, SeqCst);
        o_queue.queue_order.store(v, SeqCst);
    });
    encoder.clear_buffer(&buffer, 0, None);

    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(order.counter.load(SeqCst), 2);
    assert_eq!(order.map_order.load(SeqCst), 0);
    assert_eq!(order.queue_order.load(SeqCst), 1);
}

/// Multiple on_submitted_work_done callbacks registered on encoder all fire after submit.
#[test]
fn encoder_multiple_on_submitted_callbacks_fire() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = make_read_buffer(&device, 4);

    let counter = Arc::new(AtomicU32::new(0));
    let c1 = Arc::clone(&counter);
    let c2 = Arc::clone(&counter);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.on_submitted_work_done(move || {
        c1.fetch_add(1, SeqCst);
    });
    encoder.on_submitted_work_done(move || {
        c2.fetch_add(1, SeqCst);
    });
    encoder.clear_buffer(&buffer, 0, None);

    queue.submit([encoder.finish()]);
    _ = device.poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(counter.load(SeqCst), 2);
}
