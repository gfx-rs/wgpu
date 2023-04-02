#[path = "../../tests/common/mod.rs"]
mod common;

use std::sync::Arc;

use super::*;
use common::{initialize_test, TestParameters};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_compute_1() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::TIMESTAMP_QUERY)
            .specific_failure(None, None, Some("V3D"), true),
        |ctx| {
            let input = &[1, 2, 3, 4];

            pollster::block_on(assert_execute_gpu(
                &ctx.device,
                &ctx.queue,
                input,
                &[0, 1, 7, 2],
            ));
        },
    );
}

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_compute_2() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::TIMESTAMP_QUERY)
            .specific_failure(None, None, Some("V3D"), true),
        |ctx| {
            let input = &[5, 23, 10, 9];

            pollster::block_on(assert_execute_gpu(
                &ctx.device,
                &ctx.queue,
                input,
                &[5, 15, 6, 19],
            ));
        },
    );
}

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn test_compute_overflow() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::TIMESTAMP_QUERY)
            .specific_failure(None, None, Some("V3D"), true),
        |ctx| {
            let input = &[77031, 837799, 8400511, 63728127];
            pollster::block_on(assert_execute_gpu(
                &ctx.device,
                &ctx.queue,
                input,
                &[350, 524, OVERFLOW, OVERFLOW],
            ));
        },
    );
}

#[test]
// Wasm doesn't support threads
fn test_multithreaded_compute() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::TIMESTAMP_QUERY)
            .specific_failure(None, None, Some("V3D"), true)
            // https://github.com/gfx-rs/wgpu/issues/3250
            .specific_failure(Some(wgpu::Backends::GL), None, Some("llvmpipe"), true),
        |ctx| {
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
        },
    );
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
