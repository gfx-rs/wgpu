struct Struct {
    atomic_scalar: atomic<u64>,
    atomic_arr: array<atomic<u64>, 2>,
}

@group(0) @binding(0)
var<storage, read_write> storage_atomic_scalar: atomic<u64>;
@group(0) @binding(1)
var<storage, read_write> storage_atomic_arr: array<atomic<u64>, 2>;
@group(0) @binding(2)
var<storage, read_write> storage_struct: Struct;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicMax(&storage_atomic_scalar, 1lu);
    atomicMax(&storage_atomic_arr[1], 1lu);
    atomicMax(&storage_struct.atomic_scalar, 1lu);
    atomicMax(&storage_struct.atomic_arr[1], 1lu);

    workgroupBarrier();

    atomicMin(&storage_atomic_scalar, 1lu);
    atomicMin(&storage_atomic_arr[1], 1lu);
    atomicMin(&storage_struct.atomic_scalar, 1lu);
    atomicMin(&storage_struct.atomic_arr[1], 1lu);
}
