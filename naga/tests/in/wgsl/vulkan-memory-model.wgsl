enable wgpu_memory_fence;

struct Data {
    values: array<u32>,
}

@group(0) @binding(0) @coherent
var<storage, read_write> coherent_buf: Data;

@group(0) @binding(1) @volatile
var<storage, read_write> volatile_buf: Data;

@group(0) @binding(2)
var<storage, read_write> plain_buf: Data;

@group(0) @binding(3)
var<storage, read_write> flag: atomic<u32>;

var<workgroup> shared_value: u32;

@compute @workgroup_size(64)
fn main() {
    shared_value = plain_buf.values[0];
    workgroupBarrier();
    coherent_buf.values[0] = shared_value;
    storageBarrier();
    volatile_buf.values[1] = volatile_buf.values[0];
    atomicStore(&flag, 1u);
    _ = atomicLoad(&flag);
    _ = atomicAdd(&flag, 1u);
    storageFence();
}
