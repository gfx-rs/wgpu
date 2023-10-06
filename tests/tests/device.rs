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
fn device_destroy_then_more() {
    // This is a test of device behavior after device.destroy. Specifically, all operations
    // should trigger errors since the device is lost.
    //
    // On DX12 this test fails with a validation error in the very artifical actions taken
    // after lose the device. The error is "ID3D12CommandAllocator::Reset: The command
    // allocator cannot be reset because a command list is currently being recorded with the
    // allocator." That may indicate that DX12 doesn't like opened command buffers staying
    // open even after they return an error. For now, this test is skipped on DX12.
    //
    // The DX12 issue may be related to https://github.com/gfx-rs/wgpu/issues/3193.
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE)
            .skip(FailureCase::backend(wgpu::Backends::DX12)),
        |ctx| {
            // Create some resources on the device that we will attempt to use *after* losing
            // the device.

            // Create some 512 x 512 2D textures.
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

            let texture_for_read = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: texture_extent,
                mip_level_count: 2,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Uint,
                usage: wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

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
            let buffer_for_map = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            });
            let buffer_for_unmap = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: true,
            });

            // Create a bind group layout.
            let bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[],
                    });

            // Create a shader module.
            let shader_module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("")),
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

            let mut encoder_for_buffer_buffer_copy = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let mut encoder_for_buffer_texture_copy = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let mut encoder_for_texture_buffer_copy = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let mut encoder_for_texture_texture_copy = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            // Destroy the device. This will cause all other requests to return some variation of
            // a device invalid error.
            ctx.device.destroy();

            // TODO: verify the following operations will return an invalid device error:
            // * Run a compute or render pass
            // * Finish a render bundle encoder
            // * Create a texture from HAL
            // * Create a buffer from HAL
            // * Create a sampler
            // * Validate a surface configuration
            // * Start or stop capture
            // * Get or set buffer sub data

            // TODO: figure out how to structure a test around these operations which panic when
            // the device is invalid:
            // * device.features()
            // * device.limits()
            // * device.downlevel_properties()
            // * device.create_query_set()

            // TODO: change these fail calls to check for the specific errors which indicate that
            // the device is not valid.

            // Creating a commmand encoder should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            });

            // Creating a buffer should fail.
            fail(&ctx.device, || {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 256,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
            });

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
                    },
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
                encoder_for_buffer_buffer_copy.copy_buffer_to_buffer(
                    &buffer_source,
                    0,
                    &buffer_dest,
                    0,
                    256,
                );
            });

            // Copying a buffer to a texture should fail.
            fail(&ctx.device, || {
                encoder_for_buffer_texture_copy.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer_source,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_for_write.as_image_copy(),
                    texture_extent,
                );
            });

            // Copying a texture to a buffer should fail.
            fail(&ctx.device, || {
                encoder_for_texture_buffer_copy.copy_texture_to_buffer(
                    texture_for_read.as_image_copy(),
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer_source,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_extent,
                );
            });

            // Copying a texture to a texture should fail.
            fail(&ctx.device, || {
                encoder_for_texture_texture_copy.copy_texture_to_texture(
                    texture_for_read.as_image_copy(),
                    texture_for_write.as_image_copy(),
                    texture_extent,
                );
            });

            // Creating a bind group layout should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[],
                    });
            });

            // Creating a bind group should fail.
            fail(&ctx.device, || {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            buffer_source.as_entire_buffer_binding(),
                        ),
                    }],
                });
            });

            // Creating a pipeline layout should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[],
                        push_constant_ranges: &[],
                    });
            });

            // Creating a shader module should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("")),
                    });
            });

            // Creating a shader module spirv should fail.
            fail(&ctx.device, || unsafe {
                ctx.device
                    .create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                        label: None,
                        source: std::borrow::Cow::Borrowed(&[]),
                    });
            });

            // Creating a render pipeline should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: None,
                        layout: None,
                        vertex: wgpu::VertexState {
                            module: &shader_module,
                            entry_point: "",
                            buffers: &[],
                        },
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        fragment: None,
                        multiview: None,
                    });
            });

            // Creating a compute pipeline should fail.
            fail(&ctx.device, || {
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: None,
                        module: &shader_module,
                        entry_point: "",
                    });
            });

            // Buffer map should fail.
            fail(&ctx.device, || {
                buffer_for_map
                    .slice(..)
                    .map_async(wgpu::MapMode::Write, |_| ());
            });

            // Buffer unmap should fail.
            fail(&ctx.device, || {
                buffer_for_unmap.unmap();
            });
        },
    )
}
