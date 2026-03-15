struct Data {
    values: array<u32>,
}

@group(0) @binding(0) 
@coherent var<storage, read_write> coherent_buf: Data;
@group(0) @binding(1) 
var<storage, read_write> plain_buf: Data;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e6 = plain_buf.values[0];
    coherent_buf.values[0] = _e6;
    return;
}
