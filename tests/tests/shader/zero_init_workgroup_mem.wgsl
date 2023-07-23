const array_size = 512u;

struct WStruct {
    arr: array<u32, array_size>,
    atom: atomic<u32>
}

var<workgroup> w_mem: WStruct;

@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn read(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    var is_zero = true;
    for(var i = 0u; i < array_size; i++) {
        is_zero &= w_mem.arr[i] == 0u;
    }
    is_zero &= atomicLoad(&w_mem.atom) == 0u;

    let idx = wgid.x + (wgid.y * num_workgroups.x) + (wgid.z * num_workgroups.x * num_workgroups.y);
    output[idx] = u32(!is_zero);
}

@compute @workgroup_size(1)
fn write() {
    for(var i = 0u; i < array_size; i++) {
        w_mem.arr[i] = i;
    }
    atomicStore(&w_mem.atom, 3u);
}
