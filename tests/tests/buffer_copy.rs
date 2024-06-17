//! Tests for buffer copy validation.

use wgt::BufferAddress;

use wgpu_test::{fail_if, gpu_test, GpuTestConfiguration};

fn try_copy(
    ctx: &wgpu_test::TestingContext,
    offset: BufferAddress,
    size: BufferAddress,
    should_fail: bool,
) {
    let buffer = ctx.device.create_buffer(&BUFFER_DESCRIPTOR);
    let data = vec![255; size as usize];
    fail_if(
        &ctx.device,
        should_fail,
        || ctx.queue.write_buffer(&buffer, offset, &data),
        None,
    );
}

#[gpu_test]
static COPY_ALIGNMENT: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    try_copy(&ctx, 0, 0, false);
    try_copy(&ctx, 4, 16 + 1, true);
    try_copy(&ctx, 64, 20 + 2, true);
    try_copy(&ctx, 256, 44 + 3, true);
    try_copy(&ctx, 1024, 8 + 4, false);

    try_copy(&ctx, 0, 4, false);
    try_copy(&ctx, 4 + 1, 8, true);
    try_copy(&ctx, 64 + 2, 12, true);
    try_copy(&ctx, 256 + 3, 16, true);
    try_copy(&ctx, 1024 + 4, 4, false);
});

const BUFFER_SIZE: BufferAddress = 1234;

const BUFFER_DESCRIPTOR: wgpu::BufferDescriptor = wgpu::BufferDescriptor {
    label: None,
    size: BUFFER_SIZE,
    usage: wgpu::BufferUsages::COPY_SRC.union(wgpu::BufferUsages::COPY_DST),
    mapped_at_creation: false,
};
