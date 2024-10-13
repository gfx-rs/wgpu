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

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore(&storage_atomic_scalar, 1.5);
    atomicStore(&storage_atomic_arr[1], 1.5);
    atomicStore(&storage_struct.atomic_scalar, 1.5);
    atomicStore(&storage_struct.atomic_arr[1], 1.5);

    workgroupBarrier();

    let l0 = atomicLoad(&storage_atomic_scalar);
    let l1 = atomicLoad(&storage_atomic_arr[1]);
    let l2 = atomicLoad(&storage_struct.atomic_scalar);
    let l3 = atomicLoad(&storage_struct.atomic_arr[1]);

    workgroupBarrier();

    atomicAdd(&storage_atomic_scalar, 1.5);
    atomicAdd(&storage_atomic_arr[1], 1.5);
    atomicAdd(&storage_struct.atomic_scalar, 1.5);
    atomicAdd(&storage_struct.atomic_arr[1], 1.5);

    workgroupBarrier();

    atomicExchange(&storage_atomic_scalar, 1.5);
    atomicExchange(&storage_atomic_arr[1], 1.5);
    atomicExchange(&storage_struct.atomic_scalar, 1.5);
    atomicExchange(&storage_struct.atomic_arr[1], 1.5);

    // // TODO: https://github.com/gpuweb/gpuweb/issues/2021
    // atomicCompareExchangeWeak(&storage_atomic_scalar, 1.5);
    // atomicCompareExchangeWeak(&storage_atomic_arr[1], 1.5);
    // atomicCompareExchangeWeak(&storage_struct.atomic_scalar, 1.5);
    // atomicCompareExchangeWeak(&storage_struct.atomic_arr[1], 1.5);
}
