//! Tests of [`wgpu::util`].

use nanorand::Rng;
use wgpu::BufferUsages;

/// Generate (deterministic) random staging belt operations to exercise its logic.
#[test]
fn staging_belt_random_test() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let mut rng = nanorand::WyRand::new_seed(0xDEAD_BEEF);
    let buffer_size = 1024;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let mut belt = wgpu::util::StagingBelt::new(device.clone(), buffer_size / 2);
    let target_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for _batch in 0..100 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for _write in 0..5 {
            let offset: u64 = rng.generate_range(0..=(buffer_size - align) / align) * align;
            let size: u64 = rng.generate_range(1..=(buffer_size - offset) / align) * align;
            println!("offset {offset} size {size}");

            let mut slice = belt.write_buffer(
                &mut encoder,
                &target_buffer,
                offset,
                wgpu::BufferSize::new(size).unwrap(),
            );
            slice[0] = 1; // token amount of actual writing, just in case it makes a difference
        }

        belt.finish();
        queue.submit([encoder.finish()]);
        belt.recall();
    }
}

#[test]
fn staging_belt_panics_with_invalid_buffer_usages() {
    fn test_if_panics(usage: BufferUsages) -> bool {
        std::panic::catch_unwind(|| {
            let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
            let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(device.clone(), 512, usage);
        })
        .is_err()
    }

    for mut usage in BufferUsages::all()
        .difference(BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE)
        .iter()
    {
        assert!(test_if_panics(usage), "StagingBelt::new_with_buffer_usages should panic without MAPPABLE_PRIMARY_BUFFERS with usage={usage:?}");

        usage.insert(BufferUsages::MAP_WRITE);
        assert!(test_if_panics(usage), "StagingBelt::new_with_buffer_usages should panic without MAPPABLE_PRIMARY_BUFFERS with usage={usage:?}");
    }
}

#[test]
fn staging_belt_works_with_non_exclusive_buffer_usages() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        BufferUsages::COPY_SRC,
    );
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
    );
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        BufferUsages::MAP_WRITE,
    );
}
