use wasm_bindgen_test::*;

use wgpu_test::{fail, initialize_test, FailureCase, TestParameters};

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
fn device_lose_then_more() {
    initialize_test(TestParameters::default().features(wgpu::Features::CLEAR_TEXTURE), |ctx| {
        // Create some resources on the device that we will attempt to use *after* losing the
        // device.

        // Create a 512 x 512 2D texture, and a target view of it.
        let texture_extent = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let texture_for_view = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = texture_for_view.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_for_write = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Uint,
            usage: wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create some buffers.
        let buffer_source = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_dest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create some command encoders.
        let mut encoder_for_clear = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut encoder_for_compute_pass = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut encoder_for_render_pass = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut encoder_for_buffer_copy = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Lose the device. This will cause all other requests to return some variation of a
        // device invalid error.
        ctx.device.lose();

        // TODO: change these fail calls to check for the specific errors which indicate that
        // the device is not valid.

        // Creating a texture should fail.
        fail(&ctx.device, || {
            ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 512,
                    height: 512,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 2,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Uint,
                usage: wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        });

        // Texture clear should fail.
        fail(&ctx.device, || {
            encoder_for_clear.clear_texture(
                &texture_for_write,
                &wgpu::ImageSubresourceRange {
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                }
            );
        });

        // Creating a compute pass should fail.
        fail(&ctx.device, || {
            encoder_for_compute_pass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        });

        // Creating a render pass should fail.
        fail(&ctx.device, || {
            encoder_for_render_pass.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    ops: wgpu::Operations::default(),
                    resolve_target: None,
                    view: &target_view,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        });

        // Copying a buffer to a buffer should fail.
        fail(&ctx.device, || {
            encoder_for_buffer_copy.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 256);
        });

        // Copying a buffer to a texture should fail.
        fail(&ctx.device, || {
            encoder_for_buffer_copy.copy_buffer_to_texture(wgpu::ImageCopyBuffer {
                buffer: &buffer_source,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
            }, texture_for_write.as_image_copy(), texture_extent);
        });
    })
}
