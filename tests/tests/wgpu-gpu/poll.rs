use std::{num::NonZeroU64, time::Duration};

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandBuffer,
    CommandEncoderDescriptor, ComputePassDescriptor, PollType, ShaderStages,
};

use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        WAIT,
        WAIT_WITH_TIMEOUT,
        WAIT_WITH_TIMEOUT_MAX,
        DOUBLE_WAIT,
        WAIT_ON_SUBMISSION,
        WAIT_ON_SUBMISSION_WITH_TIMEOUT,
        WAIT_ON_SUBMISSION_WITH_TIMEOUT_MAX,
        DOUBLE_WAIT_ON_SUBMISSION,
        WAIT_OUT_OF_ORDER,
        WAIT_AFTER_BAD_SUBMISSION,
        WAIT_ON_FAILED_SUBMISSION,
    ]);
}

fn generate_dummy_work(ctx: &TestingContext) -> CommandBuffer {
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 16,
        usage: BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                },
                count: None,
            }],
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(buffer.as_entire_buffer_binding()),
        }],
    });

    let mut cmd_buf = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    let mut cpass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::default());
    cpass.set_bind_group(0, &bind_group, &[]);
    drop(cpass);

    cmd_buf.finish()
}

#[gpu_test]
static WAIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_WITH_TIMEOUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(1)),
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_WITH_TIMEOUT_MAX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::MAX),
        })
        .await
        .unwrap();
    });

#[gpu_test]
static DOUBLE_WAIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .await
        .unwrap();
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_ON_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_ON_SUBMISSION_WITH_TIMEOUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(1)),
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_ON_SUBMISSION_WITH_TIMEOUT_MAX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::MAX),
        })
        .await
        .unwrap();
    });

#[gpu_test]
static DOUBLE_WAIT_ON_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index.clone()),
            timeout: None,
        })
        .await
        .unwrap();
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .await
        .unwrap();
    });

#[gpu_test]
static WAIT_OUT_OF_ORDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let cmd_buf1 = generate_dummy_work(&ctx);
        let cmd_buf2 = generate_dummy_work(&ctx);

        let index1 = ctx.queue.submit(Some(cmd_buf1));
        let index2 = ctx.queue.submit(Some(cmd_buf2));
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index2),
            timeout: None,
        })
        .await
        .unwrap();
        ctx.async_poll(PollType::Wait {
            submission_index: Some(index1),
            timeout: None,
        })
        .await
        .unwrap();
    });

/// Submit a command buffer to the wrong device. A wait poll shouldn't hang.
///
/// We can't catch panics on Wasm, since they get reported directly to the
/// console.
#[gpu_test]
static WAIT_AFTER_BAD_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        wgpu_test::TestParameters::default()
            .skip(wgpu_test::FailureCase::webgl2())
            .enable_noop(),
    )
    .run_async(wait_after_bad_submission);

async fn wait_after_bad_submission(ctx: TestingContext) {
    let (device2, queue2) =
        wgpu_test::initialize_device(&ctx.adapter, ctx.device_features, ctx.device_limits.clone())
            .await;

    let command_buffer1 = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default())
        .finish();

    // This should panic, since the command buffer belongs to the wrong
    // device, and queue submission errors seem to be fatal errors?
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        queue2.submit([command_buffer1]);
    }));
    assert!(result.is_err());

    // This should not hang.
    //
    // Specifically, the failed submission should not cause a new fence value to
    // be allocated that will not be signalled until further work is
    // successfully submitted, causing a greater fence value to be signalled.
    device2.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Wait on a submission index that corresponds to a *failed* submission,
/// where a *later* submission succeeded — i.e. the failed index is a "hole"
/// below `last_successful_submission_index`.
///
/// This bypasses the wgpu-core `maintain()` guard (which only rejects indices
/// strictly greater than `last_successful_submission_index`) and exercises a
/// corner case of the HAL `wait` precondition documented at
/// `wgpu-hal/src/lib.rs`:
///
/// > The `value` argument must not exceed the highest value that an actual
/// > operation you have already presented to the device is going to store in
/// > `fence`.
///
/// Regression test for <https://github.com/gfx-rs/wgpu/issues/9498>.
#[gpu_test]
static WAIT_ON_FAILED_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(wgpu_test::TestParameters::default())
    .run_async(wait_on_failed_submission);

async fn wait_on_failed_submission(ctx: TestingContext) {
    // Create an alternate device; we will produce a failed submission by
    // submitting a command buffer to the wrong device.
    let (device2, queue2) =
        wgpu_test::initialize_device(&ctx.adapter, ctx.device_features, ctx.device_limits.clone())
            .await;

    // 1. Successful empty submit. Advances `last_successful_submission_index`.
    let _idx_before = queue2.submit([]);

    // 2. Failed submit (cross-device cmd buffer). Burns an index but does not
    //    advance `last_successful_submission_index`. Capture the returned
    //    index using an error scope so the validation error is not fatal.
    let command_buffer_wrong_device = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default())
        .finish();
    let scope = device2.push_error_scope(wgpu::ErrorFilter::Validation);
    let bad_index = queue2.submit([command_buffer_wrong_device]);
    let scope_error = scope.pop().await;
    assert!(
        scope_error.is_some(),
        "expected the cross-device submission to produce a validation error"
    );

    // 3. Another successful submit, this time with substantial work, so the
    //    GPU is still busy when we issue the poll below.
    //
    //    Several backends' HAL `wait` implementations have a fast path "if
    //    last_completed >= wait_value, return immediately". Per-submit
    //    bookkeeping (e.g. GLES `Fence::get_latest`) probes pending fences
    //    for completion and advances `last_completed`, so an empty submit
    //    here would let the fast path apply and mask the code path we
    //    want to exercise — the pending-fence search where the only
    //    candidate fence has a value strictly greater than `wait_value`.
    let big_size = 64 * 1024 * 1024;
    let src = device2.create_buffer(&BufferDescriptor {
        label: Some("src"),
        size: big_size,
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dst = device2.create_buffer(&BufferDescriptor {
        label: Some("dst"),
        size: big_size,
        usage: BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device2.create_command_encoder(&CommandEncoderDescriptor::default());
    // Stack several copies to extend the GPU work duration.
    for _ in 0..16 {
        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, Some(big_size));
    }
    let _idx_after = queue2.submit([encoder.finish()]);

    // 4. The interesting wait: pass the hole index to `poll(Wait)`. The
    //    wgpu-core guard sees `bad_index <= last_successful_submission_index`
    //    and lets it through to the HAL `wait`, which is being asked to wait
    //    on a fence value no operation actually presented. Must not panic or
    //    hang.
    let result = device2.poll(wgpu::PollType::Wait {
        submission_index: Some(bad_index),
        timeout: Some(Duration::from_secs(5)),
    });
    let _ = result;
}
