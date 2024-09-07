struct Struct {
    atomic_scalar: atomic<f32>,
    atomic_arr: array<atomic<f32>, 2>,
}

@group(0) @binding(0) 
var<storage, read_write> storage_atomic_scalar: atomic<f32>;
@group(0) @binding(1) 
var<storage, read_write> storage_atomic_arr: array<atomic<f32>, 2>;
@group(0) @binding(2) 
var<storage, read_write> storage_struct: Struct;

@compute @workgroup_size(2, 1, 1) 
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore((&storage_atomic_scalar), 1f);
    atomicStore((&storage_atomic_arr[1]), 1f);
    atomicStore((&storage_struct.atomic_scalar), 1f);
    atomicStore((&storage_struct.atomic_arr[1]), 1f);
    workgroupBarrier();
    let l0_ = atomicLoad((&storage_atomic_scalar));
    let l1_ = atomicLoad((&storage_atomic_arr[1]));
    let l2_ = atomicLoad((&storage_struct.atomic_scalar));
    let l3_ = atomicLoad((&storage_struct.atomic_arr[1]));
    workgroupBarrier();
    let _e27 = atomicAdd((&storage_atomic_scalar), 1f);
    let _e31 = atomicAdd((&storage_atomic_arr[1]), 1f);
    let _e35 = atomicAdd((&storage_struct.atomic_scalar), 1f);
    let _e40 = atomicAdd((&storage_struct.atomic_arr[1]), 1f);
    workgroupBarrier();
    let _e43 = atomicSub((&storage_atomic_scalar), 1f);
    let _e47 = atomicSub((&storage_atomic_arr[1]), 1f);
    let _e51 = atomicSub((&storage_struct.atomic_scalar), 1f);
    let _e56 = atomicSub((&storage_struct.atomic_arr[1]), 1f);
    workgroupBarrier();
    let _e59 = atomicExchange((&storage_atomic_scalar), 1f);
    let _e63 = atomicExchange((&storage_atomic_arr[1]), 1f);
    let _e67 = atomicExchange((&storage_struct.atomic_scalar), 1f);
    let _e72 = atomicExchange((&storage_struct.atomic_arr[1]), 1f);
    return;
}
