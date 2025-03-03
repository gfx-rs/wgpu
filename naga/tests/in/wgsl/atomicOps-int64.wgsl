// This test covers the cross product of:
//
// * All int64 atomic operations.
// * On all applicable scopes (storage read-write, workgroup).
// * For all shapes of modeling atomic data.

struct Struct {
    atomic_scalar: atomic<u64>,
    atomic_arr: array<atomic<i64>, 2>,
}

@group(0) @binding(0)
var<storage, read_write> storage_atomic_scalar: atomic<u64>;
@group(0) @binding(1)
var<storage, read_write> storage_atomic_arr: array<atomic<i64>, 2>;
@group(0) @binding(2)
var<storage, read_write> storage_struct: Struct;

var<workgroup> workgroup_atomic_scalar: atomic<u64>;
var<workgroup> workgroup_atomic_arr: array<atomic<i64>, 2>;
var<workgroup> workgroup_struct: Struct;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore(&storage_atomic_scalar, 1lu);
    atomicStore(&storage_atomic_arr[1], 1li);
    atomicStore(&storage_struct.atomic_scalar, 1lu);
    atomicStore(&storage_struct.atomic_arr[1], 1li);
    atomicStore(&workgroup_atomic_scalar, 1lu);
    atomicStore(&workgroup_atomic_arr[1], 1li);
    atomicStore(&workgroup_struct.atomic_scalar, 1lu);
    atomicStore(&workgroup_struct.atomic_arr[1], 1li);

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

    atomicAdd(&storage_atomic_scalar, 1lu);
    atomicAdd(&storage_atomic_arr[1], 1li);
    atomicAdd(&storage_struct.atomic_scalar, 1lu);
    atomicAdd(&storage_struct.atomic_arr[1], 1li);
    atomicAdd(&workgroup_atomic_scalar, 1lu);
    atomicAdd(&workgroup_atomic_arr[1], 1li);
    atomicAdd(&workgroup_struct.atomic_scalar, 1lu);
    atomicAdd(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicSub(&storage_atomic_scalar, 1lu);
    atomicSub(&storage_atomic_arr[1], 1li);
    atomicSub(&storage_struct.atomic_scalar, 1lu);
    atomicSub(&storage_struct.atomic_arr[1], 1li);
    atomicSub(&workgroup_atomic_scalar, 1lu);
    atomicSub(&workgroup_atomic_arr[1], 1li);
    atomicSub(&workgroup_struct.atomic_scalar, 1lu);
    atomicSub(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicMax(&storage_atomic_scalar, 1lu);
    atomicMax(&storage_atomic_arr[1], 1li);
    atomicMax(&storage_struct.atomic_scalar, 1lu);
    atomicMax(&storage_struct.atomic_arr[1], 1li);
    atomicMax(&workgroup_atomic_scalar, 1lu);
    atomicMax(&workgroup_atomic_arr[1], 1li);
    atomicMax(&workgroup_struct.atomic_scalar, 1lu);
    atomicMax(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicMin(&storage_atomic_scalar, 1lu);
    atomicMin(&storage_atomic_arr[1], 1li);
    atomicMin(&storage_struct.atomic_scalar, 1lu);
    atomicMin(&storage_struct.atomic_arr[1], 1li);
    atomicMin(&workgroup_atomic_scalar, 1lu);
    atomicMin(&workgroup_atomic_arr[1], 1li);
    atomicMin(&workgroup_struct.atomic_scalar, 1lu);
    atomicMin(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicAnd(&storage_atomic_scalar, 1lu);
    atomicAnd(&storage_atomic_arr[1], 1li);
    atomicAnd(&storage_struct.atomic_scalar, 1lu);
    atomicAnd(&storage_struct.atomic_arr[1], 1li);
    atomicAnd(&workgroup_atomic_scalar, 1lu);
    atomicAnd(&workgroup_atomic_arr[1], 1li);
    atomicAnd(&workgroup_struct.atomic_scalar, 1lu);
    atomicAnd(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicOr(&storage_atomic_scalar, 1lu);
    atomicOr(&storage_atomic_arr[1], 1li);
    atomicOr(&storage_struct.atomic_scalar, 1lu);
    atomicOr(&storage_struct.atomic_arr[1], 1li);
    atomicOr(&workgroup_atomic_scalar, 1lu);
    atomicOr(&workgroup_atomic_arr[1], 1li);
    atomicOr(&workgroup_struct.atomic_scalar, 1lu);
    atomicOr(&workgroup_struct.atomic_arr[1], 1li);

    workgroupBarrier();

    atomicXor(&storage_atomic_scalar, 1lu);
    atomicXor(&storage_atomic_arr[1], 1li);
    atomicXor(&storage_struct.atomic_scalar, 1lu);
    atomicXor(&storage_struct.atomic_arr[1], 1li);
    atomicXor(&workgroup_atomic_scalar, 1lu);
    atomicXor(&workgroup_atomic_arr[1], 1li);
    atomicXor(&workgroup_struct.atomic_scalar, 1lu);
    atomicXor(&workgroup_struct.atomic_arr[1], 1li);

    atomicExchange(&storage_atomic_scalar, 1lu);
    atomicExchange(&storage_atomic_arr[1], 1li);
    atomicExchange(&storage_struct.atomic_scalar, 1lu);
    atomicExchange(&storage_struct.atomic_arr[1], 1li);
    atomicExchange(&workgroup_atomic_scalar, 1lu);
    atomicExchange(&workgroup_atomic_arr[1], 1li);
    atomicExchange(&workgroup_struct.atomic_scalar, 1lu);
    atomicExchange(&workgroup_struct.atomic_arr[1], 1li);

    // // TODO: https://github.com/gpuweb/gpuweb/issues/2021
    // atomicCompareExchangeWeak(&storage_atomic_scalar, 1lu);
    // atomicCompareExchangeWeak(&storage_atomic_arr[1], 1li);
    // atomicCompareExchangeWeak(&storage_struct.atomic_scalar, 1lu);
    // atomicCompareExchangeWeak(&storage_struct.atomic_arr[1], 1li);
    // atomicCompareExchangeWeak(&workgroup_atomic_scalar, 1lu);
    // atomicCompareExchangeWeak(&workgroup_atomic_arr[1], 1li);
    // atomicCompareExchangeWeak(&workgroup_struct.atomic_scalar, 1lu);
    // atomicCompareExchangeWeak(&workgroup_struct.atomic_arr[1], 1li);
}
