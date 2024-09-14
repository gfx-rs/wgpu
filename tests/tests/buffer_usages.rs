//! Tests for buffer usages validation.

use wgpu::{BufferUsages as Bu, MapMode as Ma};
use wgpu_test::{fail_if, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};
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
    // these two require acceleration_structures feature
    Bu::all().intersection(Bu::BLAS_INPUT.union(Bu::TLAS_INPUT).complement()),
];
const INVALID_BITS: Bu = Bu::from_bits_retain(0b1111111111111);
const ALWAYS_FAIL: &[Bu; 2] = &[Bu::empty(), INVALID_BITS];

fn try_create(ctx: TestingContext, usages: &[(bool, &[wgpu::BufferUsages])]) {
    for (expect_validation_error, usage) in usages
        .iter()
        .flat_map(|&(expect_error, usages)| usages.iter().copied().map(move |u| (expect_error, u)))
    {
        fail_if(
            &ctx.device,
            expect_validation_error,
            || {
                let _buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: BUFFER_SIZE,
                    usage,
                    mapped_at_creation: false,
                });
            },
            None,
        );
    }
}

#[gpu_test]
static BUFFER_USAGE: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    try_create(
        ctx,
        &[
            (false, ALWAYS_VALID),
            (true, NEEDS_MAPPABLE_PRIMARY_BUFFERS),
            (true, ALWAYS_FAIL),
        ],
    );
});

#[gpu_test]
static BUFFER_USAGE_MAPPABLE_PRIMARY_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS))
    .run_sync(|ctx| {
        try_create(
            ctx,
            &[
                (false, ALWAYS_VALID),
                (false, NEEDS_MAPPABLE_PRIMARY_BUFFERS),
                (true, ALWAYS_FAIL),
            ],
        );
    });

async fn map_test(
    ctx: &TestingContext,
    usage_type: &str,
    map_mode_type: Ma,
    before_unmap: bool,
    before_destroy: bool,
    after_unmap: bool,
    after_destroy: bool,
) {
    log::info!("map_test usage_type:{usage_type} map_mode_type:{:?} before_unmap:{before_unmap} before_destroy:{before_destroy} after_unmap:{after_unmap} after_destroy:{after_destroy}", map_mode_type);

    let size = 8;
    let usage = match usage_type {
        "read" => Bu::COPY_DST | Bu::MAP_READ,
        "write" => Bu::COPY_SRC | Bu::MAP_WRITE,
        _ => Bu::from_bits(0).unwrap(),
    };
    let buffer_creation_validation_error = usage.is_empty();

    let mut buffer = None;

    fail_if(
        &ctx.device,
        buffer_creation_validation_error,
        || {
            buffer = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            }));
        },
        None,
    );
    if buffer_creation_validation_error {
        return;
    }

    let buffer = buffer.unwrap();

    let map_async_validation_error = buffer_creation_validation_error
        || (map_mode_type == Ma::Read && !usage.contains(Bu::MAP_READ))
        || (map_mode_type == Ma::Write && !usage.contains(Bu::MAP_WRITE));

    fail_if(
        &ctx.device,
        map_async_validation_error,
        || {
            buffer.slice(0..size).map_async(map_mode_type, |_| {});
        },
        None,
    );

    if map_async_validation_error {
        return;
    }

    if before_unmap {
        buffer.unmap();
    }

    if before_destroy {
        buffer.destroy();
    }

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    if !before_unmap && !before_destroy {
        {
            let view = buffer.slice(0..size).get_mapped_range();
            assert!(!view.is_empty());
        }

        if after_unmap {
            buffer.unmap();
        }

        if after_destroy {
            buffer.destroy();
        }
    }
}

#[gpu_test]
static BUFFER_MAP_ASYNC_MAP_STATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS))
    .run_async(move |ctx| async move {
        for usage_type in ["invalid", "read", "write"] {
            for map_mode_type in [Ma::Read, Ma::Write] {
                for before_unmap in [false, true] {
                    for before_destroy in [false, true] {
                        for after_unmap in [false, true] {
                            for after_destroy in [false, true] {
                                map_test(
                                    &ctx,
                                    usage_type,
                                    map_mode_type,
                                    before_unmap,
                                    before_destroy,
                                    after_unmap,
                                    after_destroy,
                                )
                                .await
                            }
                        }
                    }
                }
            }
        }
    });
