enable wgpu_memory_fence;

struct Data {
    values: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> payload: Data;

@group(0) @binding(1)
var<storage, read_write> flag: atomic<u32>;

var<workgroup> stage: u32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) index: u32) {
    // Producer: publish the payload, then signal through the flag.
    payload.values[index] = index;
    storageFence();
    if (index == 0u) {
        atomicStore(&flag, 1u);
    }

    // Consumer: a fence needs no uniform control flow, so every invocation
    // may spin and acquire independently.
    var spins = 0u;
    loop {
        if (atomicLoad(&flag) != 0u) {
            storageFence();
            stage = payload.values[0];
            workgroupFence();
            break;
        }
        spins += 1u;
        if (spins > 65536u) {
            break;
        }
    }
}
