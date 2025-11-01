//! Tests of [`wgpu::util`].

use nanorand::Rng;

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
