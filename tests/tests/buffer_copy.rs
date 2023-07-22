//! Tests for buffer copy validation.

use wasm_bindgen_test::wasm_bindgen_test;
use wgt::BufferAddress;

use wgpu_test::{fail_if, initialize_test, TestParameters};

#[test]
#[wasm_bindgen_test]
fn copy_alignment() {
    fn try_copy(offset: BufferAddress, size: BufferAddress, should_fail: bool) {
        initialize_test(TestParameters::default(), |ctx| {
            let buffer = ctx.device.create_buffer(&BUFFER_DESCRIPTOR);
            let data = vec![255; size as usize];
            fail_if(&ctx.device, should_fail, || {
                ctx.queue.write_buffer(&buffer, offset, &data)
            });
        });
    }

    try_copy(0, 0, false);
    try_copy(4, 16 + 1, true);
    try_copy(64, 20 + 2, true);
    try_copy(256, 44 + 3, true);
    try_copy(1024, 8 + 4, false);

    try_copy(0, 4, false);
    try_copy(4 + 1, 8, true);
    try_copy(64 + 2, 12, true);
    try_copy(256 + 3, 16, true);
    try_copy(1024 + 4, 4, false);
}

const BUFFER_SIZE: BufferAddress = 1234;

const BUFFER_DESCRIPTOR: wgpu::BufferDescriptor = wgpu::BufferDescriptor {
    label: None,
    size: BUFFER_SIZE,
    usage: wgpu::BufferUsages::COPY_SRC.union(wgpu::BufferUsages::COPY_DST),
    mapped_at_creation: false,
};
