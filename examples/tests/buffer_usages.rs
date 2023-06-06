//! Tests for buffer usages validation.

use crate::common::{fail_if, initialize_test, TestParameters};
use wasm_bindgen_test::*;
use wgt::BufferAddress;

const BUFFER_SIZE: BufferAddress = 1234;

#[test]
#[wasm_bindgen_test]
fn buffer_usage() {
    fn try_create(enable_mappable_primary_buffers: bool, usages: &[(bool, &[wgpu::BufferUsages])]) {
        let mut parameters = TestParameters::default();
        if enable_mappable_primary_buffers {
            parameters = parameters.features(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        }

        initialize_test(parameters, |ctx| {
            for (expect_validation_error, usage) in
                usages.iter().flat_map(|&(expect_error, usages)| {
                    usages.iter().copied().map(move |u| (expect_error, u))
                })
            {
                fail_if(&ctx.device, expect_validation_error, || {
                    let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: BUFFER_SIZE,
                        usage,
                        mapped_at_creation: false,
                    });
                });
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
    let invalid_bits = Bu::from_bits_retain(0b1111111111111);
    let always_fail = &[Bu::empty(), invalid_bits];

    try_create(
        false,
        &[
            (false, always_valid),
            (true, needs_mappable_primary_buffers),
            (true, always_fail),
        ],
    );
    try_create(
        true, // enable Features::MAPPABLE_PRIMARY_BUFFERS
        &[
            (false, always_valid),
            (false, needs_mappable_primary_buffers),
            (true, always_fail),
        ],
    );
}
