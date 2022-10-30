//! Tests for buffer usages validation.

use crate::common::{fail, initialize_test, valid, TestParameters};
use wgt::BufferAddress;

const BUFFER_SIZE: BufferAddress = 1234;

#[test]
fn buffer_usage() {
    fn try_create(enable_mappable_primary_buffers: bool, usages: &[(bool, &[wgpu::BufferUsages])]) {
        let mut parameters = TestParameters::default();
        if enable_mappable_primary_buffers {
            parameters = parameters.features(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        }

        initialize_test(parameters, |ctx| {
            for (expect_validation_error, usage) in
                usages
                    .iter()
                    .flat_map(|&(expect_validation_error, usages)| {
                        usages
                            .iter()
                            .copied()
                            .map(move |u| (expect_validation_error, u))
                    })
            {
                let create_buffer = || {
                    let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: BUFFER_SIZE,
                        usage,
                        mapped_at_creation: false,
                    });
                };
                if expect_validation_error {
                    fail(&ctx.device, create_buffer);
                } else {
                    valid(&ctx.device, create_buffer);
                }
            }
        });
    }

    use wgpu::BufferUsages as Bu;

    let always_valid = &[
        Bu::MAP_READ,
        Bu::MAP_WRITE,
        Bu::MAP_READ | Bu::COPY_DST,
        Bu::MAP_WRITE | Bu::COPY_SRC,
    ];
    // MAP_READ can only be paired with COPY_DST and MAP_WRITE can only be paired with COPY_SRC
    // (unless Features::MAPPABlE_PRIMARY_BUFFERS is enabled).
    let needs_mappable_primary_buffers = &[
        Bu::MAP_READ | Bu::COPY_DST | Bu::COPY_SRC,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::COPY_DST,
        Bu::MAP_READ | Bu::MAP_WRITE,
        Bu::MAP_WRITE | Bu::MAP_READ,
        Bu::MAP_READ | Bu::COPY_DST | Bu::STORAGE,
        Bu::MAP_WRITE | Bu::COPY_SRC | Bu::STORAGE,
        Bu::all(),
    ];
    let always_fail = &[Bu::empty()];

    try_create(
        false,
        &[
            (false, always_valid),
            (true, needs_mappable_primary_buffers),
            (true, always_fail),
        ],
    );
    try_create(
        true,
        &[
            (false, always_valid),
            (false, needs_mappable_primary_buffers),
            (true, always_fail),
        ],
    );
}
