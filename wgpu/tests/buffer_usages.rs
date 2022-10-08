//! Tests for buffer usages validation.

use wgt::BufferAddress;

use crate::common::{initialize_test, TestParameters};

#[test]
fn buffer_usage() {
    fn try_create(
        usages: &[wgpu::BufferUsages],
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
            for usage in usages.iter().copied() {
                let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: BUFFER_SIZE,
                    usage,
                    mapped_at_creation: false,
                });
            }
        });
    }

    use wgpu::BufferUsages as Bu;

    let always_valid = [
        Bu::MAP_READ,
        Bu::MAP_WRITE,
        Bu::MAP_READ | Bu::COPY_DST,
        Bu::MAP_WRITE | Bu::COPY_SRC,
    ];
    // MAP_READ can only be paired with COPY_DST and MAP_WRITE can only be paired with COPY_SRC
    // (unless Features::MAPPABlE_PRIMARY_BUFFERS is enabled).
    let needs_mappable_primary_buffers = [
        Bu::MAP_READ | Bu::COPY_DST | Bu::COPY_SRC,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::COPY_DST,
        Bu::MAP_READ | Bu::MAP_WRITE,
        Bu::MAP_WRITE | Bu::MAP_READ,
        Bu::MAP_READ | Bu::COPY_DST | Bu::STORAGE,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::STORAGE,
        wgpu::BufferUsages::all(),
    ];
    let always_fail = [Bu::empty()];

    try_create(&always_valid, false, false);
    try_create(&always_valid, true, false);

    try_create(&needs_mappable_primary_buffers, false, true);
    try_create(&needs_mappable_primary_buffers, true, false);

    try_create(&always_fail, false, true);
    try_create(&always_fail, true, true);
}

const BUFFER_SIZE: BufferAddress = 1234;
