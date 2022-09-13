//! Tests for buffer usages validation.

use wgt::BufferAddress;

use crate::common::{initialize_test, TestParameters};

#[test]
fn buffer_usage() {
    fn try_create(
        usage: wgpu::BufferUsages,
        enable_mappable_primary_buffers: bool,
        should_panic: bool,
    ) {
        let mut parameters = TestParameters::default();
        if enable_mappable_primary_buffers {
            parameters = parameters.features(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        }
        if should_panic {
            parameters = parameters.failure();
        }

        initialize_test(parameters, |ctx| {
            let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: BUFFER_SIZE,
                usage,
                mapped_at_creation: false,
            });
        });
    }

    use wgpu::BufferUsages as Bu;

    // These are always valid
    [
        Bu::MAP_READ,
        Bu::MAP_WRITE,
        Bu::MAP_READ | Bu::COPY_DST,
        Bu::MAP_WRITE | Bu::COPY_SRC,
    ]
    .into_iter()
    .for_each(|usages| {
        // Should only pass validation when MAPPABLE_PRIMARY_BUFFERS is enabled.
        try_create(usages, false, false);
        try_create(usages, true, false);
    });

    // MAP_READ can only be paired with COPY_DST and MAP_WRITE can only be paired with COPY_SRC
    // (unless Features::MAPPABlE_PRIMARY_BUFFERS is enabled).
    [
        Bu::MAP_READ | Bu::COPY_DST | Bu::COPY_SRC,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::COPY_DST,
        Bu::MAP_READ | Bu::MAP_WRITE,
        Bu::MAP_WRITE | Bu::MAP_READ,
        Bu::MAP_READ | Bu::COPY_DST | Bu::STORAGE,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::STORAGE,
        wgpu::BufferUsages::all(),
    ]
    .into_iter()
    .for_each(|usages| {
        // Only valid when MAPPABLE_PRIMARY_BUFFERS is enabled.
        try_create(usages, false, true);
        try_create(usages, true, false);
    });

    // Buffers cannot have empty usage flags
    [Bu::empty()].into_iter().for_each(|usages| {
        // Should always fail
        try_create(usages, false, true);
        try_create(usages, true, true);
    });
}

const BUFFER_SIZE: BufferAddress = 1234;
