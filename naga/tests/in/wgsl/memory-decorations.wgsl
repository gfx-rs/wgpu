struct Data {
    values: array<u32>,
}

@group(0) @binding(0) @coherent
var<storage, read_write> coherent_buf: Data;

@group(0) @binding(1) @volatile
var<storage, read_write> volatile_buf: Data;

@group(0) @binding(2) @coherent @volatile
var<storage, read_write> both_buf: Data;

@group(0) @binding(3)
var<storage, read_write> plain_buf: Data;

@compute @workgroup_size(1)
fn main() {
    coherent_buf.values[0] = volatile_buf.values[0];
    both_buf.values[0] = plain_buf.values[0];
}
