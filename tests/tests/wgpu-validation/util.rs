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
            // token amount of actual writing, just in case it makes a difference
            slice.slice(..1).copy_from_slice(&[1]);
        }

        belt.finish();
        queue.submit([encoder.finish()]);
        belt.recall();
    }
}

#[test]
fn staging_belt_panics_with_invalid_buffer_usages() {
    #[track_caller]
    fn test_if_panics(usage: wgpu::BufferUsages) {
        let result = std::panic::catch_unwind(|| {
            let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
            let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(device.clone(), 512, usage);
        });

        if let Err(panic) = result {
            // according to [1] the panic payload is either a `&str` or `String`
            // [1]: https://doc.rust-lang.org/std/macro.panic.html

            let message = if let Some(message) = panic.downcast_ref::<&str>() {
                *message
            } else if let Some(message) = panic.downcast_ref::<String>() {
                message.as_str()
            } else {
                // don't know what this panic is, but it's not ours
                std::panic::resume_unwind(panic);
            };

            let expected_message = format!("Only BufferUsages::COPY_SRC may be used when Features::MAPPABLE_PRIMARY_BUFFERS is not enabled. Specified buffer usages: {usage:?}");
            if expected_message == message {
                // panicked with the correct message
            } else {
                // This is not our panic (or the panic message was changed)
                std::panic::resume_unwind(panic);
            }
        } else {
            panic!("StagingBelt::new_with_buffer_usages should panic without MAPPABLE_PRIMARY_BUFFERS with usage={usage:?}");
        }
    }

    // This tests that `StagingBelt::new_with_buffer_usages` panics for any buffer usages that contain anything else than `COPY_SRC | MAP_WRITE`.
    //
    // First we iterate over all possible buffer usages except `COPY_SRC | MAP_WRITE` (anything invalid).
    for mut usage in wgpu::BufferUsages::all()
        .difference(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE)
        .iter()
    {
        // check if the constructor panics with the selected buffer usage
        test_if_panics(usage);

        // add MAP_WRITE to the selected buffer usage and check that the constructor still panics
        usage.insert(wgpu::BufferUsages::MAP_WRITE);
        test_if_panics(usage);
    }
}

#[test]
fn staging_belt_works_with_non_exclusive_buffer_usages() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        wgpu::BufferUsages::COPY_SRC,
    );
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
    );
    let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        wgpu::BufferUsages::MAP_WRITE,
    );
}

#[test]
fn staging_belt_works_with_exclusive_buffer_usages_with_mappable_primary_buffers() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
        ..Default::default()
    });

    // This tests that `StagingBelt::new_with_buffer_usages` works for any buffer usages that contain anything else than `COPY_SRC | MAP_WRITE`.
    //
    // First we iterate over all possible buffer usages except `COPY_SRC | MAP_WRITE` (anything that would be invalid without mappable primary buffers).
    for usage in wgpu::BufferUsages::all()
        .difference(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE)
        .iter()
    {
        // Check that the constructor doesn't panic without explicit `MAP_WRITE`
        let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(device.clone(), 512, usage);

        // Check that the constructor doesn't panic with explicitly `MAP_WRITE`
        let _belt = wgpu::util::StagingBelt::new_with_buffer_usages(
            device.clone(),
            512,
            wgpu::BufferUsages::MAP_WRITE,
        );
    }
}
