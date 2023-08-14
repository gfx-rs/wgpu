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
#[cfg(not(target_arch = "wasm32"))]
fn extra_limits() {
    let max_buffer_size = 8192;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        extra_limits: Some(Box::new(wgpu::Limits {
            max_buffer_size,
            ..Default::default()
        })),
        ..Default::default()
    });

    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        assert!(
            adapter.limits().max_buffer_size <= max_buffer_size,
            "adapter.max_buffer_size {:?} should be <= {max_buffer_size}",
            adapter.limits().max_buffer_size
        );
    }

    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("No suitable GPU adapters found on the system!");

    assert!(adapter.limits().max_buffer_size <= max_buffer_size);

    let should_fail = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::default(),
            limits: wgpu::Limits {
                max_buffer_size: max_buffer_size + 1,
                ..Default::default()
            },
        },
        None,
    ));

    assert!(should_fail.is_err());

    let (device, _) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::default(),
            limits: wgpu::Limits {
                max_buffer_size,
                ..Default::default()
            },
        },
        None,
    ))
    .unwrap();

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 9000,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
    }))
    .is_err();

    let validation_failed = pollster::block_on(device.pop_error_scope()).is_some();

    assert!(!panicked);
    assert!(validation_failed);
}
