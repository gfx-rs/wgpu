var<workgroup> wg_scalar: atomic<u32>;
var<workgroup> wg_signed: atomic<i32>;

@compute @workgroup_size(64, 1, 1) 
fn test_atomic_workgroup_uniform_load(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    var local: bool;

    let active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    let _e11 = atomicOr((&wg_scalar), u32((active_tile_index >= 64u)));
    let _e14 = atomicAdd((&wg_signed), 1i);
    workgroupBarrier();
    let _e16 = workgroupUniformLoad((&wg_scalar));
    let _e18 = workgroupUniformLoad((&wg_signed));
    if (_e16 == 0u) {
        local = (_e18 > 0i);
    } else {
        local = false;
    }
    let _e26 = local;
    if _e26 {
        return;
    } else {
        return;
    }
}
