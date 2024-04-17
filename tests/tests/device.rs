use std::sync::atomic::AtomicBool;

use wgpu_test::{fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static CROSS_DEVICE_BIND_GROUP_USAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::always()))
    .run_async(|ctx| async move {
        // Create a bind group using a layout from another device. This should be a validation
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

        ctx.async_poll(wgpu::Maintain::Poll)
            .await
            .panic_on_timeout();
    });

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[gpu_test]
static DEVICE_LIFETIME_CHECK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|_| {
        use pollster::FutureExt as _;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
            flags: wgpu::InstanceFlags::advanced_debugging().with_env(),
        });

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .block_on()
            .expect("failed to create adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .block_on()
            .expect("failed to create device");

        instance.poll_all(false);

        let pre_report = instance.generate_report().unwrap();

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
    .run_sync(|_| {
        use pollster::FutureExt as _;

        fn create_device_and_queue() -> (wgpu::Device, wgpu::Queue) {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all()),
                dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env()
                    .unwrap_or_default(),
                gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
                flags: wgpu::InstanceFlags::advanced_debugging().with_env(),
            });

            let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
                .block_on()
                .expect("failed to create adapter");

            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .block_on()
                .expect("failed to create device")
        }

        let _ = vec![create_device_and_queue(), create_device_and_queue()];
    });

#[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
#[gpu_test]
static REQUEST_DEVICE_ERROR_MESSAGE_NATIVE: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|_ctx| request_device_error_message());

/// Check that `RequestDeviceError`s produced have some diagnostic information.
///
/// Note: this is a wasm *and* native test. On wasm it is run directly; on native, indirectly
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
async fn request_device_error_message() {
    // Not using initialize_test() because that doesn't let us catch the error
    // nor .await anything
    let (_instance, adapter, _surface_guard) = wgpu_test::initialize_adapter(0).await;

    let device_error = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
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

// This is a test of device behavior after device.destroy. Specifically, all operations
// should trigger errors since the device is lost.
//
// On DX12 this test fails with a validation error in the very artificial actions taken
// after lose the device. The error is "ID3D12CommandAllocator::Reset: The command
// allocator cannot be reset because a command list is currently being recorded with the
// allocator." That may indicate that DX12 doesn't like opened command buffers staying
// open even after they return an error. For now, this test is skipped on DX12.
//
// The DX12 issue may be related to https://github.com/gfx-rs/wgpu/issues/3193.
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

        // Creating a command encoder should fail.
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
                        compilation_options: Default::default(),
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
                    compilation_options: Default::default(),
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
                matches!(reason, wgt::DeviceLostReason::Destroyed),
                "Device lost info reason should match DeviceLostReason::Destroyed."
            );
        });
        ctx.device.set_device_lost_callback(callback);

        // Destroy the device.
        ctx.device.destroy();

        // Make sure the device queues are empty, which ensures that the closure
        // has been called.
        assert!(ctx
            .async_poll(wgpu::Maintain::wait())
            .await
            .is_queue_empty());

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[gpu_test]
static DEVICE_DROP_THEN_LOST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
    .run_sync(|ctx| {
        // This test checks that when the device is dropped (such as in a GC),
        // the provided DeviceLostClosure is called with reason DeviceLostReason::Unknown.
        // Fails on webgl because webgl doesn't implement drop.
        static WAS_CALLED: std::sync::atomic::AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, message| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(reason, wgt::DeviceLostReason::Dropped);
            assert_eq!(message, "Device dropped.");
        });
        ctx.device.set_device_lost_callback(callback);

        // Drop the device.
        drop(ctx.device);

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[gpu_test]
static DEVICE_INVALID_THEN_SET_LOST_CALLBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
    .run_sync(|ctx| {
        // This test checks that when the device is invalid, a subsequent call
        // to set the device lost callback will immediately call the callback.
        // Invalidating the device is done via a testing-only method. Fails on
        // webgl because webgl doesn't implement make_invalid.

        // Make the device invalid.
        ctx.device.make_invalid();

        static WAS_CALLED: AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, _m| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(reason, wgt::DeviceLostReason::DeviceInvalid);
        });
        ctx.device.set_device_lost_callback(callback);

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[gpu_test]
static DEVICE_LOST_REPLACED_CALLBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        // This test checks that a device_lost_callback is called when it is
        // replaced by another callback.
        static WAS_CALLED: AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, _m| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(reason, wgt::DeviceLostReason::ReplacedCallback);
        });
        ctx.device.set_device_lost_callback(callback);

        // Replace the callback.
        let replacement_callback = Box::new(move |_r, _m| {});
        ctx.device.set_device_lost_callback(replacement_callback);

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[gpu_test]
static DROPPED_GLOBAL_THEN_DEVICE_LOST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().skip(FailureCase::always()))
    .run_sync(|ctx| {
        // What we want to do is to drop the Global, forcing a code path that
        // eventually calls Device.prepare_to_die, without having first dropped
        // the device. This models what might happen in a user agent that kills
        // wgpu without providing a more orderly shutdown. In such a case, the
        // device lost callback should be invoked with the message "Device is
        // dying."
        static WAS_CALLED: AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, message| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(reason, wgt::DeviceLostReason::Dropped);
            assert_eq!(message, "Device is dying.");
        });
        ctx.device.set_device_lost_callback(callback);

        // TODO: Drop the Global, somehow.

        // Confirm that the callback was invoked.
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

        let trivial_shaders_with_some_reversed_bindings = "\
@group(0) @binding(3) var myTexture2: texture_2d<f32>;
@group(0) @binding(2) var myTexture1: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {
  return textureSample(myTexture1, mySampler, pos.xy) + textureSample(myTexture2, mySampler, pos.xy);
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
";

        let trivial_shaders_with_some_reversed_bindings =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        trivial_shaders_with_some_reversed_bindings.into(),
                    ),
                });

        let my_texture = ctx.device.create_texture(&wgt::TextureDescriptor {
            label: None,
            size: wgt::Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format: wgt::TextureFormat::Rgba8Unorm,
            usage: wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let my_texture_view = my_texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: wgt::TextureAspect::All,
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
                    entry_point: "fs_main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgt::ColorTargetState {
                        format: wgt::TextureFormat::Bgra8Unorm,
                        blend: None,
                        write_mask: wgt::ColorWrites::ALL,
                    })],
                }),
                layout: None,

                // Other fields below aren't interesting for this text.
                label: None,
                vertex: wgpu::VertexState {
                    module: &trivial_shaders_with_some_reversed_bindings,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgt::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgt::MultisampleState::default(),
                multiview: None,
            });

        // fail(&ctx.device, || {
        // }, "");
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
