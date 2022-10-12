use crate::common::{initialize_test, TestParameters, TestingContext};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

fn test_empty_buffer_range(ctx: &TestingContext, buffer_size: u64, label: &str) {
    let status = Arc::new(AtomicBool::new(false));

    let r = wgpu::BufferUsages::MAP_READ;
    let rw = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
    for usage in [r, rw] {
        let b0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let done = status.clone();
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_ok());
            done.store(true, Ordering::SeqCst);
        });

        while !status.load(Ordering::SeqCst) {
            ctx.device.poll(wgpu::MaintainBase::Poll);
        }

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
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_err());
        });
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_err());
        });
        b0.slice(0..0).map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_err());
        });
        b0.unmap();

        status.store(false, Ordering::SeqCst);

        // Write mode.
        if usage == rw {
            let done = status.clone();
            b0.slice(0..0).map_async(wgpu::MapMode::Write, move |result| {
                assert!(result.is_ok());
                done.store(true, Ordering::SeqCst);
            });

            while !status.load(Ordering::SeqCst) {
                ctx.device.poll(wgpu::MaintainBase::Poll);
            }

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

    for _ in  0..10 {
        ctx.device.poll(wgpu::MaintainBase::Poll);
    }
}

#[test]
fn empty_buffer() {
    initialize_test(
        TestParameters::default(),
        |ctx| {
            test_empty_buffer_range(&ctx, 2048, "regular buffer");
            test_empty_buffer_range(&ctx, 0, "zero-sized buffer");
        }
    )
}
