struct testBufferBlock {
    data: array<u32>,
}

struct testBufferWriteOnlyBlock {
    data: array<u32>,
}

struct testBufferReadOnlyBlock {
    data: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> testBuffer: testBufferBlock;
@group(0) @binding(1) 
var<storage, read_write> testBufferWriteOnly: testBufferWriteOnlyBlock;
@group(0) @binding(2) 
var<storage> testBufferReadOnly: testBufferReadOnlyBlock;

fn main_1() {
    var a: u32;
    var b: u32;

    let _e12 = testBuffer.data[0];
    a = _e12;
    testBuffer.data[1] = u32(2);
    testBufferWriteOnly.data[1] = u32(2);
    let _e27 = testBufferReadOnly.data[0];
    b = _e27;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
