use std::sync::atomic::AtomicBool;

use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static CROSS_DEVICE_BIND_GROUP_USAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::always()))
    .run_async(|ctx| async move {
        // Create a bind group using a layout from another device. This should be a validation
        // error but currently crashes.
        let (device2, _) =
            pollster::block_on(ctx.adapter.request_device(&Default::default())).unwrap();

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

        ctx.async_poll(wgpu::PollType::Poll).await.unwrap();
    });

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[gpu_test]
static DEVICE_LIFETIME_CHECK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        ctx.instance.poll_all(false);

        let pre_report = ctx.instance.generate_report().unwrap();

        let TestingContext {
            instance,
            device,
            queue,
            ..
        } = ctx;

        drop(queue);
        drop(device);

        let post_report = instance.generate_report().unwrap();

        assert_ne!(
            pre_report, post_report,
            "Queue and Device has not been dropped as expected"
        );
    });

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[gpu_test]
static MULTIPLE_DEVICES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        use pollster::FutureExt as _;
        ctx.adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .block_on()
            .expect("failed to create device");
        ctx.adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .block_on()
            .expect("failed to create device");
    });

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[gpu_test]
static REQUEST_DEVICE_ERROR_MESSAGE_NATIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters({
        let default = TestParameters::default();

        // On CI, we have the vulkan SDK installed and this test initializes the Vulkan backend when all
        // normal tests do not, so we get these weird error messages. This only actually gets hooked up
        // on Windows, because that's the only platform where the actual runtime gets hooked up and there
        // are no drivers.
        if std::env::var("WGPU_CI").is_ok() && cfg!(windows) {
            default.expect_fail(
                FailureCase::always()
                    .validation_error("Registry lookup failed to get ICD manifest files.  Possibly missing Vulkan driver?")
                    .validation_error("vkCreateInstance: Found no drivers!")
            )
        } else {
            default
        }
    })
    .run_async(|_ctx| request_device_error_message());

/// Check that `RequestDeviceError`s produced have some diagnostic information.
///
/// Note: this is a wasm *and* native test. On wasm it is run directly; on native, indirectly
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
async fn request_device_error_message() {
    // Not using initialize_test() because that doesn't let us catch the error
    // nor .await anything
    let (_instance, adapter, _surface_guard) =
        wgpu_test::initialize_adapter(None, &TestParameters::default()).await;

    let device_error = adapter
        .request_device(&wgpu::DeviceDescriptor {
            // Force a failure by requesting absurd limits.
            required_features: wgpu::Features::all(),
            required_limits: wgpu::Limits {
                max_texture_dimension_1d: u32::MAX,
                max_texture_dimension_2d: u32::MAX,
                max_texture_dimension_3d: u32::MAX,
                max_bind_groups: u32::MAX,
                max_push_constant_size: u32::MAX,
                ..Default::default()
            },
            ..Default::default()
        })
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
            let expected = "Unsupported features were requested: Features {";
        }
    }
    assert!(device_error.contains(expected), "{device_error}");
}

// This is a test of device behavior after `device.destroy()`. Specifically, all operations
// should turn into no-ops, per spec.
#[gpu_test]
static DEVICE_DESTROY_THEN_MORE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::CLEAR_TEXTURE))
    .run_sync(|ctx| {
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
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_for_unmap = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
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

        // Destroy the device. This will cause all further operations to become no-ops.
        ctx.device.destroy();

        // The following operations should fail if the device was not lost.
        // Since the device is lost, we check that they succeed.

        let _ = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ctx.device.limits().max_buffer_size * 2,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let _ = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: ctx.device.limits().max_texture_dimension_2d * 2,
                height: ctx.device.limits().max_texture_dimension_2d * 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Uint,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        encoder_for_clear.clear_texture(
            &texture_for_write,
            &wgpu::ImageSubresourceRange {
                aspect: wgpu::TextureAspect::StencilOnly, // texture has no stencil component
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            },
        );
        ctx.queue.submit([encoder_for_clear.finish()]);

        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Occlusion,
            count: u32::MAX, // can be at most 4096
        });

        let pass = encoder_for_compute_pass.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: None,
            }),
        });
        drop(pass);
        ctx.queue.submit([encoder_for_compute_pass.finish()]);

        let pass = encoder_for_render_pass.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &target_view,
                depth_slice: Some(u32::MAX), // should not be provided for 2D views
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        drop(pass);
        ctx.queue.submit([encoder_for_render_pass.finish()]);

        encoder_for_buffer_buffer_copy.copy_buffer_to_buffer(
            &buffer_source,
            0,
            &buffer_dest,
            0,
            u64::MAX, // out of bounds for both buffers
        );
        ctx.queue.submit([encoder_for_buffer_buffer_copy.finish()]);

        encoder_for_buffer_texture_copy.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer_source,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: u64::MAX, // out of bounds for buffer
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
            },
            texture_for_write.as_image_copy(),
            texture_extent,
        );
        ctx.queue.submit([encoder_for_buffer_texture_copy.finish()]);

        encoder_for_texture_buffer_copy.copy_texture_to_buffer(
            texture_for_read.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer_source,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: u64::MAX, // out of bounds for buffer
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );
        ctx.queue.submit([encoder_for_texture_buffer_copy.finish()]);

        encoder_for_texture_texture_copy.copy_texture_to_texture(
            texture_for_read.as_image_copy(),
            texture_for_write.as_image_copy(),
            wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: u32::MAX, // out of bounds for both textures
            },
        );
        ctx.queue
            .submit([encoder_for_texture_texture_copy.finish()]);

        let invalid_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::all(),
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(
                                (ctx.device.limits().max_uniform_buffer_binding_size * 2) as u64,
                            ),
                        },
                        count: None,
                    }],
                });

        let _ = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &invalid_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(buffer_source.as_entire_buffer_binding()),
            }],
        });

        let invalid_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&invalid_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let _ = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    "not a valid shader source",
                )),
            });

        let _ = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&invalid_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some(""),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: None,
                multiview: None,
                cache: None,
            });

        let _ = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&invalid_pipeline_layout),
                module: &shader_module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        buffer_for_map
            .slice(..)
            .map_async(wgpu::MapMode::Write, |_| ());

        buffer_for_unmap.unmap();
    });

#[gpu_test]
static DEVICE_DESTROY_THEN_LOST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        // This test checks that when device.destroy is called, the provided
        // DeviceLostClosure is called with reason DeviceLostReason::Destroyed.
        static WAS_CALLED: AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, _m| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert!(
                matches!(reason, wgpu::DeviceLostReason::Destroyed),
                "Device lost info reason should match DeviceLostReason::Destroyed."
            );
        });
        ctx.device.set_device_lost_callback(callback);

        // Destroy the device.
        ctx.device.destroy();

        // Make sure the device queues are empty, which ensures that the closure
        // has been called.
        assert!(ctx
            .async_poll(wgpu::PollType::wait())
            .await
            .unwrap()
            .is_queue_empty());

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[gpu_test]
static DIFFERENT_BGL_ORDER_BW_SHADER_AND_API: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        // This test addresses a bug found in multiple backends where `wgpu_core` and `wgpu_hal`
        // backends made different assumptions about the element order of vectors of bind group
        // layout entries and bind group resource bindings.
        //
        // Said bug was exposed originally by:
        //
        // 1. Shader-declared bindings having a different order than resource bindings provided to
        //    `Device::create_bind_group`.
        // 2. Having more of one type of resource in the bind group than another.
        //
        // …such that internals would accidentally attempt to use an out-of-bounds index (of one
        // resource type) in the wrong list of a different resource type. Let's reproduce that
        // here.

        let trivial_shaders_with_some_reversed_bindings = concat!(
            "@group(0) @binding(3) var myTexture2: texture_2d<f32>;\n",
            "@group(0) @binding(2) var myTexture1: texture_2d<f32>;\n",
            "@group(0) @binding(1) var mySampler: sampler;\n",
            "\n",
            "@fragment\n",
            "fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {\n",
            "  return textureSample(myTexture1, mySampler, pos.xy) \n",
            "    + textureSample(myTexture2, mySampler, pos.xy);\n",
            "}\n",
            "\n",
            "@vertex\n",
            "fn vs_main() -> @builtin(position) vec4<f32> {\n",
            "  return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n",
            "}\n",
        );

        let trivial_shaders_with_some_reversed_bindings =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        trivial_shaders_with_some_reversed_bindings.into(),
                    ),
                });

        let my_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let my_texture_view = my_texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let my_sampler = ctx
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                fragment: Some(wgpu::FragmentState {
                    module: &trivial_shaders_with_some_reversed_bindings,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                layout: None,

                // Other fields below aren't interesting for this text.
                label: None,
                vertex: wgpu::VertexState {
                    module: &trivial_shaders_with_some_reversed_bindings,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // fail(&ctx.device, || {
        // }, "");
        let _ = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&my_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&my_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&my_texture_view),
                },
            ],
        });
    });

#[gpu_test]
static DEVICE_DESTROY_THEN_BUFFER_CLEANUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        // When a device is destroyed, its resources should be released,
        // without causing a deadlock.

        // Create a buffer to be left around until the device is destroyed.
        let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create a texture to be left around until the device is destroyed.
        let texture_extent = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let _texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Destroy the device.
        ctx.device.destroy();

        // Poll the device, which should try to clean up its resources.
        ctx.instance.poll_all(true);
    });

#[gpu_test]
static DEVICE_AND_QUEUE_HAVE_DIFFERENT_IDS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        let TestingContext {
            adapter,
            device_features,
            device_limits,
            device,
            queue,
            ..
        } = ctx;

        drop(device);

        let (device2, queue2) =
            wgpu_test::initialize_device(&adapter, device_features, device_limits).await;

        drop(queue);
        drop(device2);
        drop(queue2); // this would previously panic since we would try to use the Device ID to drop the Queue
    });
