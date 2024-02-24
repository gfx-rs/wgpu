use std::sync::Arc;

use parking_lot::Mutex;
use wgpu_test::{gpu_test, GpuTestConfiguration};

use wgpu::*;

/// The WebGPU specification has very specific requirements about the ordering of map_async
/// and on_submitted_work_done callbacks. Specifically, all map_async callbacks that are initiated
/// before a given on_submitted_work_done callback must be invoked before the on_submitted_work_done
/// callback is invoked.
///
/// We previously immediately invoked on_submitted_work_done callbacks if there was no active submission
/// to add them to. This is incorrect, as we do not immediately invoke map_async callbacks.
#[gpu_test]
static QUEUE_SUBMITTED_CALLBACK_ORDERING: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        // Create a mappable buffer
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("mappable buffer"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode some work using it. The specifics of this work don't matter, just
        // that the buffer is used.
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        encoder.clear_buffer(&buffer, 0, None);

        // Submit the work.
        ctx.queue.submit(Some(encoder.finish()));
        // Ensure the work is finished.
        ctx.async_poll(Maintain::wait()).await.panic_on_timeout();

        #[derive(Debug)]
        struct OrderingContext {
            /// Incremented every time a callback in invoked.
            /// This allows the callbacks to know their ordering.
            counter: u8,
            /// The value of the counter when the map_async callback was invoked.
            value_read_map_async: Option<u8>,
            /// The value of the counter when the queue submitted work done callback was invoked.
            value_read_queue_submitted: Option<u8>,
        }

        // Create shared ownership of the ordering context, and clone 2 copies.
        let ordering = Arc::new(Mutex::new(OrderingContext {
            counter: 0,
            value_read_map_async: None,
            value_read_queue_submitted: None,
        }));
        let ordering_clone_map_async = Arc::clone(&ordering);
        let ordering_clone_queue_submitted = Arc::clone(&ordering);

        // Register the callabacks.
        buffer.slice(..).map_async(MapMode::Read, move |_| {
            let mut guard = ordering_clone_map_async.lock();
            guard.value_read_map_async = Some(guard.counter);
            guard.counter += 1;
        });

        // If the bug is present, this callback will be invoked immediately inside this function,
        // despite the fact there is an outstanding map_async callback.
        ctx.queue.on_submitted_work_done(move || {
            let mut guard = ordering_clone_queue_submitted.lock();
            guard.value_read_queue_submitted = Some(guard.counter);
            guard.counter += 1;
        });

        // No GPU work is happening at this point, but we want to process callbacks.
        ctx.async_poll(MaintainBase::Poll).await.panic_on_timeout();

        // Extract the ordering out of the arc.
        let ordering = Arc::into_inner(ordering).unwrap().into_inner();

        // There were two callbacks invoked
        assert_eq!(ordering.counter, 2);
        // The map async callback was invoked fist
        assert_eq!(ordering.value_read_map_async, Some(0));
        // The queue submitted work done callback was invoked second.
        assert_eq!(ordering.value_read_queue_submitted, Some(1));
    });
