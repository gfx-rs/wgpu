use std::num::NonZeroU64;

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandBuffer,
    CommandEncoderDescriptor, ComputePassDescriptor, Maintain, ShaderStages,
};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestingContext};

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
static WAIT: GpuTestConfiguration = GpuTestConfiguration::new().run_async(|ctx| async move {
    let cmd_buf = generate_dummy_work(&ctx);

    ctx.queue.submit(Some(cmd_buf));
    ctx.async_poll(Maintain::wait()).await.panic_on_timeout();
});

#[gpu_test]
static DOUBLE_WAIT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(Maintain::wait()).await.panic_on_timeout();
        ctx.async_poll(Maintain::wait()).await.panic_on_timeout();
    });

#[gpu_test]
static WAIT_ON_SUBMISSION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(Maintain::wait_for(index))
            .await
            .panic_on_timeout();
    });

#[gpu_test]
static DOUBLE_WAIT_ON_SUBMISSION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.async_poll(Maintain::wait_for(index.clone()))
            .await
            .panic_on_timeout();
        ctx.async_poll(Maintain::wait_for(index))
            .await
            .panic_on_timeout();
    });

#[gpu_test]
static WAIT_OUT_OF_ORDER: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let cmd_buf1 = generate_dummy_work(&ctx);
        let cmd_buf2 = generate_dummy_work(&ctx);

        let index1 = ctx.queue.submit(Some(cmd_buf1));
        let index2 = ctx.queue.submit(Some(cmd_buf2));
        ctx.async_poll(Maintain::wait_for(index2))
            .await
            .panic_on_timeout();
        ctx.async_poll(Maintain::wait_for(index1))
            .await
            .panic_on_timeout();
    });

/// Submit a command buffer to the wrong device. A wait poll shouldn't hang.
///
/// We can't catch panics on Wasm, since they get reported directly to the
/// console.
#[gpu_test]
static WAIT_AFTER_BAD_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(wgpu_test::TestParameters::default().skip(wgpu_test::FailureCase::webgl2()))
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
    device2.poll(wgpu::Maintain::Wait);
}
