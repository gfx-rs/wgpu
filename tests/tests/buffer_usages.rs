//! Tests for buffer usages validation.

use wgpu::BufferUsages as Bu;
use wgpu_test::{fail_if, infra::GpuTest, TestParameters};
use wgt::BufferAddress;

const BUFFER_SIZE: BufferAddress = 1234;

const ALWAYS_VALID: &[Bu; 4] = &[
    Bu::MAP_READ,
    Bu::MAP_WRITE,
    Bu::MAP_READ.union(Bu::COPY_DST),
    Bu::MAP_WRITE.union(Bu::COPY_SRC),
];
// MAP_READ can only be paired with COPY_DST and MAP_WRITE can only be paired with COPY_SRC
// (unless Features::MAPPABlE_PRIMARY_BUFFERS is enabled).
const NEEDS_MAPPABLE_PRIMARY_BUFFERS: &[Bu; 7] = &[
    Bu::MAP_READ.union(Bu::COPY_DST.union(Bu::COPY_SRC)),
    Bu::MAP_WRITE.union(Bu::COPY_SRC.union(Bu::COPY_DST)),
    Bu::MAP_READ.union(Bu::MAP_WRITE),
    Bu::MAP_WRITE.union(Bu::MAP_READ),
    Bu::MAP_READ.union(Bu::COPY_DST.union(Bu::STORAGE)),
    Bu::MAP_WRITE.union(Bu::COPY_SRC.union(Bu::STORAGE)),
    Bu::all(),
];
const INVALID_BITS: Bu = Bu::from_bits_retain(0b1111111111111);
const ALWAYS_FAIL: &[Bu; 2] = &[Bu::empty(), INVALID_BITS];

fn try_create(ctx: wgpu_test::TestingContext, usages: &[(bool, &[wgpu::BufferUsages])]) {
    for (expect_validation_error, usage) in usages
        .iter()
        .flat_map(|&(expect_error, usages)| usages.iter().copied().map(move |u| (expect_error, u)))
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
}

#[derive(Default)]
pub struct BufferUsageTest;

impl GpuTest for BufferUsageTest {
    fn run(&self, ctx: wgpu_test::TestingContext) {
        try_create(
            ctx,
            &[
                (false, ALWAYS_VALID),
                (true, NEEDS_MAPPABLE_PRIMARY_BUFFERS),
                (true, ALWAYS_FAIL),
            ],
        );
    }
}

#[derive(Default)]
pub struct BufferUsageMappablePrimaryTest;

impl GpuTest for BufferUsageMappablePrimaryTest {
    fn parameters(&self, params: TestParameters) -> TestParameters {
        params.features(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
    }

    fn run(&self, ctx: wgpu_test::TestingContext) {
        try_create(
            ctx,
            &[
                (false, ALWAYS_VALID),
                (false, NEEDS_MAPPABLE_PRIMARY_BUFFERS),
                (true, ALWAYS_FAIL),
            ],
        );
    }
}
