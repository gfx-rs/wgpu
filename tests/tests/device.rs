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
