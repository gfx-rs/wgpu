struct testBufferBlock {
    data: array<u32>,
}

struct testBufferReadOnlyBlock {
    data: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> testBuffer: testBufferBlock;
@group(0) @binding(2) 
var<storage> testBufferReadOnly: testBufferReadOnlyBlock;

fn main_1() {
    var a: u32;
    var b: u32;

    let _e4 = testBuffer.data[0];
    a = _e4;
    testBuffer.data[1i] = 2u;
    let _e12 = testBufferReadOnly.data[0];
    b = _e12;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
