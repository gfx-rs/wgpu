use super::*;
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static COMPUTE_1: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::adapter("V3D")),
    )
    .run_async(|ctx| {
        let input = &[1, 2, 3, 4];

        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input, &[0, 1, 7, 2]).await }
    });

#[gpu_test]
static COMPUTE_2: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::adapter("V3D")),
    )
    .run_async(|ctx| {
        let input = &[5, 23, 10, 9];

        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input, &[5, 15, 6, 19]).await }
    });

#[gpu_test]
static COMPUTE_OVERFLOW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::adapter("V3D")),
    )
    .run_async(|ctx| {
        let input = &[77031, 837799, 8400511, 63728127];
        async move {
            assert_execute_gpu(
                &ctx.device,
                &ctx.queue,
                input,
                &[350, 524, OVERFLOW, OVERFLOW],
            )
            .await
        }
    });

#[cfg(not(target_arch = "wasm32"))]
#[gpu_test]
static MULTITHREADED_COMPUTE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::adapter("V3D")),
    )
    .run_sync(|ctx| {
        use std::{sync::mpsc, sync::Arc, thread, time::Duration};

        let ctx = Arc::new(ctx);

        let thread_count = 8;

        let (tx, rx) = mpsc::channel();
        let workers: Vec<_> = (0..thread_count)
            .map(move |_| {
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
                })
            })
            .collect();

        for _ in 0..thread_count {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("A thread never completed.");
        }

        for worker in workers {
            worker.join().unwrap();
        }
    });

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
