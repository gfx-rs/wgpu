enable wgpu_memory_fence;

struct Data {
    values: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> payload: Data;
@group(0) @binding(1) 
var<storage, read_write> flag: atomic<u32>;
var<workgroup> stage: u32;

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(local_invocation_index) index: u32) {
    var spins: u32 = 0u;

    payload.values[index] = index;
    storageFence();
    if (index == 0u) {
        atomicStore((&flag), 1u);
    }
    loop {
        let _e11 = atomicLoad((&flag));
        if (_e11 != 0u) {
            storageFence();
            let _e18 = payload.values[0];
            stage = _e18;
            workgroupFence();
            break;
        }
        let _e19 = spins;
        spins = (_e19 + 1u);
        let _e22 = spins;
        if (_e22 > 65536u) {
            break;
        }
    }
    return;
}
