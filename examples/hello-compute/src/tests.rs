use std::sync::Arc;

use super::*;
use wgpu_test::{infra::GpuTest, TestParameters};

#[derive(Default)]
pub struct Compute1Test;

impl GpuTest for Compute1Test {
    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .specific_failure(None, None, Some("V3D"), true)
    }

    fn run(&self, ctx: wgpu_test::TestingContext) {
        let input = &[1, 2, 3, 4];

        pollster::block_on(assert_execute_gpu(
            &ctx.device,
            &ctx.queue,
            input,
            &[0, 1, 7, 2],
        ));
    }
}

#[derive(Default)]
pub struct Compute2Test;

impl GpuTest for Compute2Test {
    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .specific_failure(None, None, Some("V3D"), true)
    }

    fn run(&self, ctx: wgpu_test::TestingContext) {
        let input = &[5, 23, 10, 9];

        pollster::block_on(assert_execute_gpu(
            &ctx.device,
            &ctx.queue,
            input,
            &[5, 15, 6, 19],
        ));
    }
}

#[derive(Default)]
pub struct ComputeOverflowTest;

impl GpuTest for ComputeOverflowTest {
    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .specific_failure(None, None, Some("V3D"), true)
    }

    fn run(&self, ctx: wgpu_test::TestingContext) {
        let input = &[77031, 837799, 8400511, 63728127];
        pollster::block_on(assert_execute_gpu(
            &ctx.device,
            &ctx.queue,
            input,
            &[350, 524, OVERFLOW, OVERFLOW],
        ));
    }
}

#[derive(Default)]
pub struct MultithreadedComputeTest;

impl GpuTest for MultithreadedComputeTest {
    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .specific_failure(None, None, Some("V3D"), true)
            // https://github.com/gfx-rs/wgpu/issues/3250
            .specific_failure(Some(wgpu::Backends::GL), None, Some("llvmpipe"), true)
    }

    fn run(&self, ctx: wgpu_test::TestingContext) {
        use std::{sync::mpsc, thread, time::Duration};

        let ctx = Arc::new(ctx);

        let thread_count = 8;

        let (tx, rx) = mpsc::channel();
        for _ in 0..thread_count {
            let tx = tx.clone();
            let ctx = Arc::clone(&ctx);
            thread::spawn(move || {
                let input = &[100, 100, 100];
                pollster::block_on(assert_execute_gpu(
                    &ctx.device,
                    &ctx.queue,
                    input,
                    &[25, 25, 25],
                ));
                tx.send(true).unwrap();
            });
        }

        for _ in 0..thread_count {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("A thread never completed.");
        }
    }
}

async fn assert_execute_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &[u32],
    expected: &[u32],
) {
    if let Some(produced) = execute_gpu_inner(device, queue, input).await {
        assert_eq!(produced, expected);
    }
}
