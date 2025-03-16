//! Tests of [`wgpu::Backend::Noop`].

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

#[test]
fn device_is_not_available_by_default() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        ..Default::default()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect_err("noop backend adapter present when it should not be");
}

#[test]
fn device_is_available_when_requested() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be");
}

#[test]
fn device_and_buffers() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    // Demonstrate that creating and *writing* to a buffer succeeds.
    // This also involves creation of a staging buffer.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hello world"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    assert_eq!(buffer.size(), 8);
    queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]);
    queue.write_buffer(&buffer, 4, &[5, 6, 7, 8]);

    // Demonstrate that we can read back data from the buffer.
    // This also involves copy_buffer_to_buffer().
    let done: Arc<AtomicBool> = Arc::default();
    let done2 = done.clone();
    wgpu::util::DownloadBuffer::read_buffer(&device, &queue, &buffer.slice(..), move |result| {
        assert_eq!(*result.unwrap(), [1, 2, 3, 4, 5, 6, 7, 8],);
        done.store(true, Relaxed);
    });
    device.poll(wgpu::PollType::Wait).unwrap();
    assert!(done2.load(Relaxed));
}
