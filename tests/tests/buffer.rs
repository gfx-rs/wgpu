use wgpu_test::{initialize_test, TestParameters, TestingContext};

fn test_empty_buffer_range(ctx: &TestingContext, buffer_size: u64, label: &str) {
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

        ctx.device.poll(wgpu::MaintainBase::Wait);

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

            ctx.device.poll(wgpu::MaintainBase::Wait);

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

    ctx.device.poll(wgpu::MaintainBase::Wait);
}

#[test]
#[ignore]
fn empty_buffer() {
    // TODO: Currently wgpu does not accept empty buffer slices, which
    // is what test is about.
    initialize_test(TestParameters::default(), |ctx| {
        test_empty_buffer_range(&ctx, 2048, "regular buffer");
        test_empty_buffer_range(&ctx, 0, "zero-sized buffer");
    })
}

#[test]
fn test_map_offset() {
    initialize_test(TestParameters::default(), |ctx| {
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

        ctx.device.poll(wgpu::MaintainBase::Wait);

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

        ctx.device.poll(wgpu::MaintainBase::Wait);

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
}
