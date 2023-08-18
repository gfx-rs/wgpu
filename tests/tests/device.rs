use wasm_bindgen_test::*;

use wgpu_test::{initialize_test, TestParameters};

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
    initialize_test(TestParameters::default().failure(), |ctx| {
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
    });
}

#[test]
fn device_lifetime_check() {
    use pollster::FutureExt as _;

    env_logger::init();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
        gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
    });

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .block_on()
        .expect("failed to create adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .block_on()
        .expect("failed to create device");

    instance.poll_all(false);

    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    let pre_report = instance.generate_report();

    drop(queue);
    drop(device);

    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    let post_report = instance.generate_report();

    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    assert_ne!(
        pre_report, post_report,
        "Queue and Device has not been dropped as expected"
    );
}
