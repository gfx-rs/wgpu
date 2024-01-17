use std::ops::Range;

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

async fn fill_test(ctx: &TestingContext, range: Range<u64>, size: u64) -> bool {
    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Initialize the whole buffer with values.
    let buffer_contents = vec![0xFF_u8; size as usize];
    ctx.queue.write_buffer(&gpu_buffer, 0, &buffer_contents);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    encoder.clear_buffer(&gpu_buffer, range.start, Some(range.end - range.start));
    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, size);

    ctx.queue.submit(Some(encoder.finish()));
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let buffer_slice = cpu_buffer.slice(..);
    let buffer_data = buffer_slice.get_mapped_range();

    let first_clear_byte = buffer_data
        .iter()
        .enumerate()
        .find_map(|(index, byte)| (*byte == 0x00).then_some(index))
        .expect("No clear happened at all");

    let first_dirty_byte = buffer_data
        .iter()
        .enumerate()
        .skip(first_clear_byte)
        .find_map(|(index, byte)| (*byte != 0x00).then_some(index))
        .unwrap_or(size as usize);

    let second_clear_byte = buffer_data
        .iter()
        .enumerate()
        .skip(first_dirty_byte)
        .find_map(|(index, byte)| (*byte == 0x00).then_some(index));

    if second_clear_byte.is_some() {
        eprintln!("Found multiple cleared ranges instead of a single clear range of {}..{} on a buffer of size {}.", range.start, range.end, size);
        return false;
    }

    let cleared_range = first_clear_byte as u64..first_dirty_byte as u64;

    if cleared_range != range {
        eprintln!(
            "Cleared range is {}..{}, but the clear range is {}..{} on a buffer of size {}.",
            cleared_range.start, cleared_range.end, range.start, range.end, size
        );
        return false;
    }

    eprintln!(
        "Cleared range is {}..{} on a buffer of size {}.",
        cleared_range.start, cleared_range.end, size
    );

    true
}

/// Nvidia has a bug in vkCmdFillBuffer where the clear range is not properly respected under
/// certain conditions. See https://github.com/gfx-rs/wgpu/issues/4122 for more information.
///
/// This test will fail on nvidia if the bug is not properly worked around.
#[gpu_test]
static CLEAR_BUFFER_RANGE_RESPECTED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        // This hits most of the cases in nvidia's clear buffer bug
        let mut succeeded = true;
        for power in 4..14 {
            let size = 1 << power;
            for start_offset in (0..=36).step_by(4) {
                for size_offset in (0..=36).step_by(4) {
                    let range = start_offset..size + size_offset + start_offset;
                    let result = fill_test(&ctx, range, 1 << 16).await;

                    succeeded &= result;
                }
            }
        }
        assert!(succeeded);
    });
