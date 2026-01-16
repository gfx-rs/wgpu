// Test workgroupUniformLoad specialization for atomic<T> -> T

struct AtomicStruct {
    atomic_scalar: atomic<u32>,
    atomic_arr: array<atomic<i32>, 2>,
}

var<workgroup> wg_scalar: atomic<u32>;
var<workgroup> wg_signed: atomic<i32>;
var<workgroup> wg_struct: AtomicStruct;

@compute @workgroup_size(64)
fn test_atomic_workgroup_uniform_load(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    let active_tile_index = workgroup_id.x + workgroup_id.y * 32768;
    
    // Each thread may set the atomics
    atomicOr(&wg_scalar, u32(active_tile_index >= 64));
    atomicAdd(&wg_signed, 1i);
    atomicStore(&wg_struct.atomic_scalar, 1u);
    atomicAdd(&wg_struct.atomic_arr[0], 1i);
    
    workgroupBarrier();
    
    // workgroupUniformLoad on atomic<u32> should return u32
    let scalar_val: u32 = workgroupUniformLoad(&wg_scalar);
    
    // workgroupUniformLoad on atomic<i32> should return i32
    let signed_val: i32 = workgroupUniformLoad(&wg_signed);
    
    // workgroupUniformLoad on struct.atomic_scalar should return u32
    let struct_scalar: u32 = workgroupUniformLoad(&wg_struct.atomic_scalar);
    
    // workgroupUniformLoad on struct.atomic_arr[i] should return i32
    let struct_arr_val: i32 = workgroupUniformLoad(&wg_struct.atomic_arr[0]);
    
    // Should be able to use all results in comparisons
    if scalar_val == 0u && signed_val > 0i && struct_scalar > 0u && struct_arr_val > 0i {
        return;
    }
}
