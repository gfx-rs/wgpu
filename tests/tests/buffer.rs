use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

async fn test_empty_buffer_range(ctx: &TestingContext, buffer_size: u64, label: &str) {
    let r = wgpu::BufferUsages::MAP_READ;
    let rw = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
    for usage in [r, rw] {
        let b0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        b0.slice(0..0)
            .map_async(wgpu::MapMode::Read, Result::unwrap);

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

        {
            let view = b0.slice(0..0).get_mapped_range();
            assert!(view.is_empty());
        }

        b0.unmap();

        // Map and unmap right away.
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |_| {});
        b0.unmap();

        // Map multiple times before unmapping.
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |_| {});
        b0.slice(0..0)
            .map_async(wgpu::MapMode::Read, move |result| {
                assert!(result.is_err());
            });
        b0.slice(0..0)
            .map_async(wgpu::MapMode::Read, move |result| {
                assert!(result.is_err());
            });
        b0.slice(0..0)
            .map_async(wgpu::MapMode::Read, move |result| {
                assert!(result.is_err());
            });
        b0.unmap();

        // Write mode.
        if usage == rw {
            b0.slice(0..0)
                .map_async(wgpu::MapMode::Write, Result::unwrap);

            ctx.async_poll(wgpu::Maintain::wait())
                .await
                .panic_on_timeout();

            //{
            //    let view = b0.slice(0..0).get_mapped_range_mut();
            //    assert!(view.is_empty());
            //}

            b0.unmap();

            // Map and unmap right away.
            b0.slice(0..0).map_async(wgpu::MapMode::Write, move |_| {});
            b0.unmap();
        }
    }

    let b1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: buffer_size,
        usage: rw,
        mapped_at_creation: true,
    });

    {
        let view = b1.slice(0..0).get_mapped_range_mut();
        assert!(view.is_empty());
    }

    b1.unmap();

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();
}

#[gpu_test]
static EMPTY_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::always()))
    .run_async(|ctx| async move {
        test_empty_buffer_range(&ctx, 2048, "regular buffer").await;
        test_empty_buffer_range(&ctx, 0, "zero-sized buffer").await;
    });

#[gpu_test]
static MAP_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new().run_async(|ctx| async move {
    // This test writes 16 bytes at the beginning of buffer mapped mapped with
    // an offset of 32 bytes. Then the buffer is copied into another buffer that
    // is read back and we check that the written bytes are correctly placed at
    // offset 32..48.
    // The goal is to check that get_mapped_range did not accidentally double-count
    // the mapped offset.

    let write_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    write_buf
        .slice(32..)
        .map_async(wgpu::MapMode::Write, move |result| {
            result.unwrap();
        });

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    {
        let slice = write_buf.slice(32..48);
        let mut view = slice.get_mapped_range_mut();
        for byte in &mut view[..] {
            *byte = 2;
        }
    }

    write_buf.unmap();

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(&write_buf, 0, &read_buf, 0, 256);

    ctx.queue.submit(Some(encoder.finish()));

    read_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, Result::unwrap);

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let slice = read_buf.slice(..);
    let view = slice.get_mapped_range();
    for byte in &view[0..32] {
        assert_eq!(*byte, 0);
    }
    for byte in &view[32..48] {
        assert_eq!(*byte, 2);
    }
    for byte in &view[48..] {
        assert_eq!(*byte, 0);
    }
});

/// The WebGPU algorithm [validating shader binding][vsb] requires
/// implementations to check that buffer bindings are large enough to
/// hold the WGSL `storage` or `uniform` variables they're bound to.
///
/// This test tries to build a pipeline from a shader module with a
/// 32-byte variable and a bindgroup layout with a min_binding_size of
/// 16 for that variable's group/index. Pipeline creation should fail.
#[gpu_test]
static MINIMUM_BUFFER_BINDING_SIZE_LAYOUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_sync(|ctx| {
        // Create a shader module that statically uses a storage buffer.
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    r#"
                        @group(0) @binding(0)
                        var<storage, read_write> a: array<u32, 8>;
                        @compute @workgroup_size(1)
                        fn main() {
                            a[0] = a[1];
                        }
            "#,
                )),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(16),
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        wgpu_test::fail(
            &ctx.device,
            || {
                let _ = ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            },
            Some("shader global resourcebinding { group: 0, binding: 0 } is not available in the pipeline layout"),
        );
    });

/// The WebGPU algorithm [validating shader binding][vsb] requires
/// implementations to check that buffer bindings are large enough to
/// hold the WGSL `storage` or `uniform` variables they're bound to.
///
/// This test tries to dispatch a compute shader that uses a 32-byte
/// variable with a bindgroup layout with a min_binding_size of zero
/// (meaning, "validate at dispatch recording time") and a 16-byte
/// binding. Command recording should fail.
#[gpu_test]
static MINIMUM_BUFFER_BINDING_SIZE_DISPATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_sync(|ctx| {
        // This test tries to use a bindgroup layout with a
        // min_binding_size of 16 to an index whose WGSL type requires 32
        // bytes. Pipeline creation should fail.

        // Create a shader module that statically uses a storage buffer.
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    r#"
                        @group(0) @binding(0)
                        var<storage, read_write> a: array<u32, 8>;
                        @compute @workgroup_size(1)
                        fn main() {
                            a[0] = a[1];
                        }
            "#,
                )),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16, // too small for 32-byte var `a` in shader module
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        wgpu_test::fail(
            &ctx.device,
            || {
                let mut encoder = ctx.device.create_command_encoder(&Default::default());

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_pipeline(&pipeline);
                pass.dispatch_workgroups(1, 1, 1);

                drop(pass);
                let _ = encoder.finish();
            },
            Some("buffer is bound with size 16 where the shader expects 32 in group[0] compact index 0"),
        );
    });

#[gpu_test]
static CLEAR_OFFSET_OUTSIDE_RESOURCE_BOUNDS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        let size = 16;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_of_bounds = size.checked_add(wgpu::COPY_BUFFER_ALIGNMENT).unwrap();

        wgpu_test::fail(
            &ctx.device,
            || {
                ctx.device
                    .create_command_encoder(&Default::default())
                    .clear_buffer(&buffer, out_of_bounds, None)
            },
            Some("Clear of 20..20 would end up overrunning the bounds of the buffer of size 16"),
        );
    });

#[gpu_test]
static CLEAR_OFFSET_PLUS_SIZE_OUTSIDE_U64_BOUNDS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_sync(|ctx| {
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16, // unimportant for this test
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let max_valid_offset = u64::MAX - (u64::MAX % wgpu::COPY_BUFFER_ALIGNMENT);
            let smallest_aligned_invalid_size = wgpu::COPY_BUFFER_ALIGNMENT;

            wgpu_test::fail(
                &ctx.device,
                || {
                    ctx.device
                        .create_command_encoder(&Default::default())
                        .clear_buffer(
                            &buffer,
                            max_valid_offset,
                            Some(smallest_aligned_invalid_size),
                        )
                },
                Some(concat!(
                    "Clear starts at offset 18446744073709551612 with size of 4, ",
                    "but these added together exceed `u64::MAX`"
                )),
            );
        });
