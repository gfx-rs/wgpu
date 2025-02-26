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
@group(0) @binding(3)
var<uniform> input: u64;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicMax(&storage_atomic_scalar, input);
    atomicMax(&storage_atomic_arr[1], 1 + input);
    atomicMax(&storage_struct.atomic_scalar, 1lu);
    atomicMax(&storage_struct.atomic_arr[1], u64(id.x));

    workgroupBarrier();

    atomicMin(&storage_atomic_scalar, input);
    atomicMin(&storage_atomic_arr[1], 1 + input);
    atomicMin(&storage_struct.atomic_scalar, 1lu);
    atomicMin(&storage_struct.atomic_arr[1], u64(id.x));
}
