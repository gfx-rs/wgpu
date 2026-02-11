struct AtomicStruct {
    atomic_scalar: atomic<u32>,
    atomic_arr: array<atomic<i32>, 2>,
}

var<workgroup> wg_scalar: atomic<u32>;
var<workgroup> wg_signed: atomic<i32>;
var<workgroup> wg_struct: AtomicStruct;

@compute @workgroup_size(64, 1, 1) 
fn test_atomic_workgroup_uniform_load(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    var local: bool;
    var local_1: bool;
    var local_2: bool;

    let active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    let _e11 = atomicOr((&wg_scalar), u32((active_tile_index >= 64u)));
    let _e14 = atomicAdd((&wg_signed), 1i);
    atomicStore((&wg_struct.atomic_scalar), 1u);
    let _e22 = atomicAdd((&wg_struct.atomic_arr[0]), 1i);
    workgroupBarrier();
    let _e24 = workgroupUniformLoad((&wg_scalar));
    let _e26 = workgroupUniformLoad((&wg_signed));
    let _e29 = workgroupUniformLoad((&wg_struct.atomic_scalar));
    let _e33 = workgroupUniformLoad((&wg_struct.atomic_arr[0]));
    if (_e24 == 0u) {
        local = (_e26 > 0i);
    } else {
        local = false;
    }
    let _e41 = local;
    if _e41 {
        local_1 = (_e29 > 0u);
    } else {
        local_1 = false;
    }
    let _e47 = local_1;
    if _e47 {
        local_2 = (_e33 > 0i);
    } else {
        local_2 = false;
    }
    let _e53 = local_2;
    if _e53 {
        return;
    } else {
        return;
    }
}
