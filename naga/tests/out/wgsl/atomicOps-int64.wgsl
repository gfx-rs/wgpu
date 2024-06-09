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

@compute @workgroup_size(2, 1, 1) 
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore((&storage_atomic_scalar), 1lu);
    atomicStore((&storage_atomic_arr[1]), 1li);
    atomicStore((&storage_struct.atomic_scalar), 1lu);
    atomicStore((&storage_struct.atomic_arr[1]), 1li);
    atomicStore((&workgroup_atomic_scalar), 1lu);
    atomicStore((&workgroup_atomic_arr[1]), 1li);
    atomicStore((&workgroup_struct.atomic_scalar), 1lu);
    atomicStore((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    let l0_ = atomicLoad((&storage_atomic_scalar));
    let l1_ = atomicLoad((&storage_atomic_arr[1]));
    let l2_ = atomicLoad((&storage_struct.atomic_scalar));
    let l3_ = atomicLoad((&storage_struct.atomic_arr[1]));
    let l4_ = atomicLoad((&workgroup_atomic_scalar));
    let l5_ = atomicLoad((&workgroup_atomic_arr[1]));
    let l6_ = atomicLoad((&workgroup_struct.atomic_scalar));
    let l7_ = atomicLoad((&workgroup_struct.atomic_arr[1]));
    workgroupBarrier();
    let _e51 = atomicAdd((&storage_atomic_scalar), 1lu);
    let _e55 = atomicAdd((&storage_atomic_arr[1]), 1li);
    let _e59 = atomicAdd((&storage_struct.atomic_scalar), 1lu);
    let _e64 = atomicAdd((&storage_struct.atomic_arr[1]), 1li);
    let _e67 = atomicAdd((&workgroup_atomic_scalar), 1lu);
    let _e71 = atomicAdd((&workgroup_atomic_arr[1]), 1li);
    let _e75 = atomicAdd((&workgroup_struct.atomic_scalar), 1lu);
    let _e80 = atomicAdd((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    let _e83 = atomicSub((&storage_atomic_scalar), 1lu);
    let _e87 = atomicSub((&storage_atomic_arr[1]), 1li);
    let _e91 = atomicSub((&storage_struct.atomic_scalar), 1lu);
    let _e96 = atomicSub((&storage_struct.atomic_arr[1]), 1li);
    let _e99 = atomicSub((&workgroup_atomic_scalar), 1lu);
    let _e103 = atomicSub((&workgroup_atomic_arr[1]), 1li);
    let _e107 = atomicSub((&workgroup_struct.atomic_scalar), 1lu);
    let _e112 = atomicSub((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    atomicMax((&storage_atomic_scalar), 1lu);
    atomicMax((&storage_atomic_arr[1]), 1li);
    atomicMax((&storage_struct.atomic_scalar), 1lu);
    atomicMax((&storage_struct.atomic_arr[1]), 1li);
    atomicMax((&workgroup_atomic_scalar), 1lu);
    atomicMax((&workgroup_atomic_arr[1]), 1li);
    atomicMax((&workgroup_struct.atomic_scalar), 1lu);
    atomicMax((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    atomicMin((&storage_atomic_scalar), 1lu);
    atomicMin((&storage_atomic_arr[1]), 1li);
    atomicMin((&storage_struct.atomic_scalar), 1lu);
    atomicMin((&storage_struct.atomic_arr[1]), 1li);
    atomicMin((&workgroup_atomic_scalar), 1lu);
    atomicMin((&workgroup_atomic_arr[1]), 1li);
    atomicMin((&workgroup_struct.atomic_scalar), 1lu);
    atomicMin((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    let _e163 = atomicAnd((&storage_atomic_scalar), 1lu);
    let _e167 = atomicAnd((&storage_atomic_arr[1]), 1li);
    let _e171 = atomicAnd((&storage_struct.atomic_scalar), 1lu);
    let _e176 = atomicAnd((&storage_struct.atomic_arr[1]), 1li);
    let _e179 = atomicAnd((&workgroup_atomic_scalar), 1lu);
    let _e183 = atomicAnd((&workgroup_atomic_arr[1]), 1li);
    let _e187 = atomicAnd((&workgroup_struct.atomic_scalar), 1lu);
    let _e192 = atomicAnd((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    let _e195 = atomicOr((&storage_atomic_scalar), 1lu);
    let _e199 = atomicOr((&storage_atomic_arr[1]), 1li);
    let _e203 = atomicOr((&storage_struct.atomic_scalar), 1lu);
    let _e208 = atomicOr((&storage_struct.atomic_arr[1]), 1li);
    let _e211 = atomicOr((&workgroup_atomic_scalar), 1lu);
    let _e215 = atomicOr((&workgroup_atomic_arr[1]), 1li);
    let _e219 = atomicOr((&workgroup_struct.atomic_scalar), 1lu);
    let _e224 = atomicOr((&workgroup_struct.atomic_arr[1]), 1li);
    workgroupBarrier();
    let _e227 = atomicXor((&storage_atomic_scalar), 1lu);
    let _e231 = atomicXor((&storage_atomic_arr[1]), 1li);
    let _e235 = atomicXor((&storage_struct.atomic_scalar), 1lu);
    let _e240 = atomicXor((&storage_struct.atomic_arr[1]), 1li);
    let _e243 = atomicXor((&workgroup_atomic_scalar), 1lu);
    let _e247 = atomicXor((&workgroup_atomic_arr[1]), 1li);
    let _e251 = atomicXor((&workgroup_struct.atomic_scalar), 1lu);
    let _e256 = atomicXor((&workgroup_struct.atomic_arr[1]), 1li);
    let _e259 = atomicExchange((&storage_atomic_scalar), 1lu);
    let _e263 = atomicExchange((&storage_atomic_arr[1]), 1li);
    let _e267 = atomicExchange((&storage_struct.atomic_scalar), 1lu);
    let _e272 = atomicExchange((&storage_struct.atomic_arr[1]), 1li);
    let _e275 = atomicExchange((&workgroup_atomic_scalar), 1lu);
    let _e279 = atomicExchange((&workgroup_atomic_arr[1]), 1li);
    let _e283 = atomicExchange((&workgroup_struct.atomic_scalar), 1lu);
    let _e288 = atomicExchange((&workgroup_struct.atomic_arr[1]), 1li);
    return;
}
