// This test covers the cross product of:
//
// * All atomic operations.
// * On all applicable scopes (storage read-write, workgroup).
// * For all shapes of modeling atomic data.

struct Struct {
    atomic_scalar: atomic<u32>,
    atomic_arr: array<atomic<i32>, 2>,
}

@group(0) @binding(0)
var<storage, read_write> storage_atomic_scalar: atomic<u32>;
@group(0) @binding(1)
var<storage, read_write> storage_atomic_arr: array<atomic<i32>, 2>;
@group(0) @binding(2)
var<storage, read_write> storage_struct: Struct;

var<workgroup> workgroup_atomic_scalar: atomic<u32>;
var<workgroup> workgroup_atomic_arr: array<atomic<i32>, 2>;
var<workgroup> workgroup_struct: Struct;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore(&storage_atomic_scalar, 1u);
    atomicStore(&storage_atomic_arr[1], 1i);
    atomicStore(&storage_struct.atomic_scalar, 1u);
    atomicStore(&storage_struct.atomic_arr[1], 1i);
    atomicStore(&workgroup_atomic_scalar, 1u);
    atomicStore(&workgroup_atomic_arr[1], 1i);
    atomicStore(&workgroup_struct.atomic_scalar, 1u);
    atomicStore(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    let l0 = atomicLoad(&storage_atomic_scalar);
    let l1 = atomicLoad(&storage_atomic_arr[1]);
    let l2 = atomicLoad(&storage_struct.atomic_scalar);
    let l3 = atomicLoad(&storage_struct.atomic_arr[1]);
    let l4 = atomicLoad(&workgroup_atomic_scalar);
    let l5 = atomicLoad(&workgroup_atomic_arr[1]);
    let l6 = atomicLoad(&workgroup_struct.atomic_scalar);
    let l7 = atomicLoad(&workgroup_struct.atomic_arr[1]);

    workgroupBarrier();

    atomicAdd(&storage_atomic_scalar, 1u);
    atomicAdd(&storage_atomic_arr[1], 1i);
    atomicAdd(&storage_struct.atomic_scalar, 1u);
    atomicAdd(&storage_struct.atomic_arr[1], 1i);
    atomicAdd(&workgroup_atomic_scalar, 1u);
    atomicAdd(&workgroup_atomic_arr[1], 1i);
    atomicAdd(&workgroup_struct.atomic_scalar, 1u);
    atomicAdd(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicSub(&storage_atomic_scalar, 1u);
    atomicSub(&storage_atomic_arr[1], 1i);
    atomicSub(&storage_struct.atomic_scalar, 1u);
    atomicSub(&storage_struct.atomic_arr[1], 1i);
    atomicSub(&workgroup_atomic_scalar, 1u);
    atomicSub(&workgroup_atomic_arr[1], 1i);
    atomicSub(&workgroup_struct.atomic_scalar, 1u);
    atomicSub(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicMax(&storage_atomic_scalar, 1u);
    atomicMax(&storage_atomic_arr[1], 1i);
    atomicMax(&storage_struct.atomic_scalar, 1u);
    atomicMax(&storage_struct.atomic_arr[1], 1i);
    atomicMax(&workgroup_atomic_scalar, 1u);
    atomicMax(&workgroup_atomic_arr[1], 1i);
    atomicMax(&workgroup_struct.atomic_scalar, 1u);
    atomicMax(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicMin(&storage_atomic_scalar, 1u);
    atomicMin(&storage_atomic_arr[1], 1i);
    atomicMin(&storage_struct.atomic_scalar, 1u);
    atomicMin(&storage_struct.atomic_arr[1], 1i);
    atomicMin(&workgroup_atomic_scalar, 1u);
    atomicMin(&workgroup_atomic_arr[1], 1i);
    atomicMin(&workgroup_struct.atomic_scalar, 1u);
    atomicMin(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicAnd(&storage_atomic_scalar, 1u);
    atomicAnd(&storage_atomic_arr[1], 1i);
    atomicAnd(&storage_struct.atomic_scalar, 1u);
    atomicAnd(&storage_struct.atomic_arr[1], 1i);
    atomicAnd(&workgroup_atomic_scalar, 1u);
    atomicAnd(&workgroup_atomic_arr[1], 1i);
    atomicAnd(&workgroup_struct.atomic_scalar, 1u);
    atomicAnd(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicOr(&storage_atomic_scalar, 1u);
    atomicOr(&storage_atomic_arr[1], 1i);
    atomicOr(&storage_struct.atomic_scalar, 1u);
    atomicOr(&storage_struct.atomic_arr[1], 1i);
    atomicOr(&workgroup_atomic_scalar, 1u);
    atomicOr(&workgroup_atomic_arr[1], 1i);
    atomicOr(&workgroup_struct.atomic_scalar, 1u);
    atomicOr(&workgroup_struct.atomic_arr[1], 1i);

    workgroupBarrier();

    atomicXor(&storage_atomic_scalar, 1u);
    atomicXor(&storage_atomic_arr[1], 1i);
    atomicXor(&storage_struct.atomic_scalar, 1u);
    atomicXor(&storage_struct.atomic_arr[1], 1i);
    atomicXor(&workgroup_atomic_scalar, 1u);
    atomicXor(&workgroup_atomic_arr[1], 1i);
    atomicXor(&workgroup_struct.atomic_scalar, 1u);
    atomicXor(&workgroup_struct.atomic_arr[1], 1i);

    atomicExchange(&storage_atomic_scalar, 1u);
    atomicExchange(&storage_atomic_arr[1], 1i);
    atomicExchange(&storage_struct.atomic_scalar, 1u);
    atomicExchange(&storage_struct.atomic_arr[1], 1i);
    atomicExchange(&workgroup_atomic_scalar, 1u);
    atomicExchange(&workgroup_atomic_arr[1], 1i);
    atomicExchange(&workgroup_struct.atomic_scalar, 1u);
    atomicExchange(&workgroup_struct.atomic_arr[1], 1i);

    // // TODO: https://github.com/gpuweb/gpuweb/issues/2021
    // atomicCompareExchangeWeak(&storage_atomic_scalar, 1u);
    // atomicCompareExchangeWeak(&storage_atomic_arr[1], 1i);
    // atomicCompareExchangeWeak(&storage_struct.atomic_scalar, 1u);
    // atomicCompareExchangeWeak(&storage_struct.atomic_arr[1], 1i);
    // atomicCompareExchangeWeak(&workgroup_atomic_scalar, 1u);
    // atomicCompareExchangeWeak(&workgroup_atomic_arr[1], 1i);
    // atomicCompareExchangeWeak(&workgroup_struct.atomic_scalar, 1u);
    // atomicCompareExchangeWeak(&workgroup_struct.atomic_arr[1], 1i);
}
