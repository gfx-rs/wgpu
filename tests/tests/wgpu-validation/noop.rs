//! Tests of [`wgpu::Backend::Noop`].

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

#[test]
fn device_is_not_available_by_default() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect_err("noop backend adapter present when it should not be");
}

#[test]
fn device_is_available_when_requested() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
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
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(done2.load(Relaxed));
}

// gfx-rs/wgpu#3794: when the instance is given a `task_executor`, native async pipeline creation
// hands the compile to it (here, on a worker thread) instead of running inline.
#[test]
fn async_pipeline_runs_on_task_executor() {
    let task_count = Arc::new(AtomicUsize::new(0));
    let ran_off_thread = Arc::new(AtomicBool::new(false));

    let executor = {
        let task_count = task_count.clone();
        let ran_off_thread = ran_off_thread.clone();
        let main_thread = std::thread::current().id();
        wgpu::TaskExecutor::new(move |task| {
            task_count.fetch_add(1, Relaxed);
            let ran_off_thread = ran_off_thread.clone();
            // Run the compile on a worker thread to demonstrate genuine off-thread dispatch, then
            // join so the result is ready by the time `execute` returns.
            std::thread::spawn(move || {
                ran_off_thread.store(std::thread::current().id() != main_thread, Relaxed);
                task.run();
            })
            .join()
            .unwrap();
        })
    };

    let instance = wgpu::Instance::new(
        wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions::enabled(),
                ..Default::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        }
        .with_task_executor(executor),
    );

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("noop device");

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trivial compute"),
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });

    let pipeline = pollster::block_on(device.create_compute_pipeline_async(
        &wgpu::ComputePipelineDescriptor {
            label: Some("async compute pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        },
    ));

    assert!(
        pipeline.is_ok(),
        "async pipeline should compile: {pipeline:?}"
    );
    assert_eq!(
        task_count.load(Relaxed),
        1,
        "the task_executor should have been handed exactly one compile task"
    );
    assert!(
        ran_off_thread.load(Relaxed),
        "the compile task should have run on the executor's worker thread"
    );
}

// gfx-rs/wgpu#3794 + the error-model decision (mirror the WebGPU spec): an invalid async pipeline
// *rejects* (returns `Err`) and does NOT additionally surface the error through the device's error
// scope — unlike the infallible sync `create_compute_pipeline`. The shader module is valid, so
// only pipeline creation (bad entry point) fails, isolating the async error path.
#[test]
fn async_pipeline_rejects_without_touching_error_scope() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trivial compute"),
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = pollster::block_on(device.create_compute_pipeline_async(
        &wgpu::ComputePipelineDescriptor {
            label: Some("async compute pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("this_entry_point_does_not_exist"),
            compilation_options: Default::default(),
            cache: None,
        },
    ));
    assert!(
        result.is_err(),
        "async pipeline creation should reject on an invalid entry point"
    );

    let scope_error = pollster::block_on(scope.pop());
    assert!(
        scope_error.is_none(),
        "async rejection must not also surface via the error scope, got: {scope_error:?}"
    );
}
