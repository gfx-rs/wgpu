use wasm_bindgen_test::*;

use wgpu_test::{initialize_test, FailureCase, TestParameters};

#[test]
#[wasm_bindgen_test]
fn device_initialization() {
    initialize_test(TestParameters::default(), |_ctx| {
        // intentionally empty
    })
}

#[test]
#[ignore]
fn device_mismatch() {
    initialize_test(
        // https://github.com/gfx-rs/wgpu/issues/3927
        TestParameters::default().expect_fail(FailureCase::always()),
        |ctx| {
            // Create a bind group uisng a lyaout from another device. This should be a validation
            // error but currently crashes.
            let (device2, _) =
                pollster::block_on(ctx.adapter.request_device(&Default::default(), None)).unwrap();
            {
                let bind_group_layout =
                    device2.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[],
                    });

                let _bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[],
                });
            }

            ctx.device.poll(wgpu::Maintain::Poll);
        },
    );
}

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[test]
fn request_device_error_on_native() {
    pollster::block_on(request_device_error_message());
}

/// Check that `RequestDeviceError`s produced have some diagnostic information.
///
/// Note: this is a wasm *and* native test. On wasm it is run directly; on native, indirectly
#[wasm_bindgen_test::wasm_bindgen_test]
async fn request_device_error_message() {
    // Not using initialize_test() because that doesn't let us catch the error
    // nor .await anything
    let (adapter, _surface_guard) = wgpu_test::initialize_adapter();

    let device_error = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                // Force a failure by requesting absurd limits.
                features: wgpu::Features::all(),
                limits: wgpu::Limits {
                    max_texture_dimension_1d: u32::MAX,
                    max_texture_dimension_2d: u32::MAX,
                    max_texture_dimension_3d: u32::MAX,
                    max_bind_groups: u32::MAX,
                    max_push_constant_size: u32::MAX,
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap_err();

    let device_error = device_error.to_string();
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))] {
            // On WebGPU, so the error we get will be from the browser WebGPU API.
            // Per the WebGPU specification this should be a `TypeError` when features are not
            // available, <https://gpuweb.github.io/gpuweb/#dom-gpuadapter-requestdevice>,
            // and the stringification it goes through for Rust should put that in the message.
            let expected = "TypeError";
        } else {
            // This message appears whenever wgpu-core is used as the implementation.
            let expected = "Unsupported features were requested: Features(";
        }
    }
    assert!(device_error.contains(expected), "{device_error}");
}

#[test]
fn device_lose() {
    initialize_test(TestParameters::default(), |_ctx| {
        //ctx.device.device_lose
    })
}