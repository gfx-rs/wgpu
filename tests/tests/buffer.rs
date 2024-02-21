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

        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        ctx.device
            .create_command_encoder(&Default::default())
            .clear_buffer(&buffer, out_of_bounds, None);
        let err_msg = pollster::block_on(ctx.device.pop_error_scope())
            .unwrap()
            .to_string();
        assert!(err_msg.contains(
            "Clear of 20..20 would end up overrunning the bounds of the buffer of size 16"
        ));
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

            ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            ctx.device
                .create_command_encoder(&Default::default())
                .clear_buffer(
                    &buffer,
                    max_valid_offset,
                    Some(smallest_aligned_invalid_size),
                );
            let err_msg = pollster::block_on(ctx.device.pop_error_scope())
                .unwrap()
                .to_string();
            assert!(err_msg.contains(concat!(
                "Clear starts at offset 18446744073709551612 with size of 4, ",
                "but these added together exceed `u64::MAX`"
            )));
        });
