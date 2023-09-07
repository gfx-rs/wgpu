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

@compute @workgroup_size(2, 1, 1) 
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    atomicStore((&storage_atomic_scalar), 1u);
    atomicStore((&storage_atomic_arr[1]), 1);
    atomicStore((&storage_struct.atomic_scalar), 1u);
    atomicStore((&storage_struct.atomic_arr[1]), 1);
    atomicStore((&workgroup_atomic_scalar), 1u);
    atomicStore((&workgroup_atomic_arr[1]), 1);
    atomicStore((&workgroup_struct.atomic_scalar), 1u);
    atomicStore((&workgroup_struct.atomic_arr[1]), 1);
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
    let _e51 = atomicAdd((&storage_atomic_scalar), 1u);
    let _e55 = atomicAdd((&storage_atomic_arr[1]), 1);
    let _e59 = atomicAdd((&storage_struct.atomic_scalar), 1u);
    let _e64 = atomicAdd((&storage_struct.atomic_arr[1]), 1);
    let _e67 = atomicAdd((&workgroup_atomic_scalar), 1u);
    let _e71 = atomicAdd((&workgroup_atomic_arr[1]), 1);
    let _e75 = atomicAdd((&workgroup_struct.atomic_scalar), 1u);
    let _e80 = atomicAdd((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e83 = atomicSub((&storage_atomic_scalar), 1u);
    let _e87 = atomicSub((&storage_atomic_arr[1]), 1);
    let _e91 = atomicSub((&storage_struct.atomic_scalar), 1u);
    let _e96 = atomicSub((&storage_struct.atomic_arr[1]), 1);
    let _e99 = atomicSub((&workgroup_atomic_scalar), 1u);
    let _e103 = atomicSub((&workgroup_atomic_arr[1]), 1);
    let _e107 = atomicSub((&workgroup_struct.atomic_scalar), 1u);
    let _e112 = atomicSub((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e115 = atomicMax((&storage_atomic_scalar), 1u);
    let _e119 = atomicMax((&storage_atomic_arr[1]), 1);
    let _e123 = atomicMax((&storage_struct.atomic_scalar), 1u);
    let _e128 = atomicMax((&storage_struct.atomic_arr[1]), 1);
    let _e131 = atomicMax((&workgroup_atomic_scalar), 1u);
    let _e135 = atomicMax((&workgroup_atomic_arr[1]), 1);
    let _e139 = atomicMax((&workgroup_struct.atomic_scalar), 1u);
    let _e144 = atomicMax((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e147 = atomicMin((&storage_atomic_scalar), 1u);
    let _e151 = atomicMin((&storage_atomic_arr[1]), 1);
    let _e155 = atomicMin((&storage_struct.atomic_scalar), 1u);
    let _e160 = atomicMin((&storage_struct.atomic_arr[1]), 1);
    let _e163 = atomicMin((&workgroup_atomic_scalar), 1u);
    let _e167 = atomicMin((&workgroup_atomic_arr[1]), 1);
    let _e171 = atomicMin((&workgroup_struct.atomic_scalar), 1u);
    let _e176 = atomicMin((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e179 = atomicAnd((&storage_atomic_scalar), 1u);
    let _e183 = atomicAnd((&storage_atomic_arr[1]), 1);
    let _e187 = atomicAnd((&storage_struct.atomic_scalar), 1u);
    let _e192 = atomicAnd((&storage_struct.atomic_arr[1]), 1);
    let _e195 = atomicAnd((&workgroup_atomic_scalar), 1u);
    let _e199 = atomicAnd((&workgroup_atomic_arr[1]), 1);
    let _e203 = atomicAnd((&workgroup_struct.atomic_scalar), 1u);
    let _e208 = atomicAnd((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e211 = atomicOr((&storage_atomic_scalar), 1u);
    let _e215 = atomicOr((&storage_atomic_arr[1]), 1);
    let _e219 = atomicOr((&storage_struct.atomic_scalar), 1u);
    let _e224 = atomicOr((&storage_struct.atomic_arr[1]), 1);
    let _e227 = atomicOr((&workgroup_atomic_scalar), 1u);
    let _e231 = atomicOr((&workgroup_atomic_arr[1]), 1);
    let _e235 = atomicOr((&workgroup_struct.atomic_scalar), 1u);
    let _e240 = atomicOr((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e243 = atomicXor((&storage_atomic_scalar), 1u);
    let _e247 = atomicXor((&storage_atomic_arr[1]), 1);
    let _e251 = atomicXor((&storage_struct.atomic_scalar), 1u);
    let _e256 = atomicXor((&storage_struct.atomic_arr[1]), 1);
    let _e259 = atomicXor((&workgroup_atomic_scalar), 1u);
    let _e263 = atomicXor((&workgroup_atomic_arr[1]), 1);
    let _e267 = atomicXor((&workgroup_struct.atomic_scalar), 1u);
    let _e272 = atomicXor((&workgroup_struct.atomic_arr[1]), 1);
    let _e275 = atomicExchange((&storage_atomic_scalar), 1u);
    let _e279 = atomicExchange((&storage_atomic_arr[1]), 1);
    let _e283 = atomicExchange((&storage_struct.atomic_scalar), 1u);
    let _e288 = atomicExchange((&storage_struct.atomic_arr[1]), 1);
    let _e291 = atomicExchange((&workgroup_atomic_scalar), 1u);
    let _e295 = atomicExchange((&workgroup_atomic_arr[1]), 1);
    let _e299 = atomicExchange((&workgroup_struct.atomic_scalar), 1u);
    let _e304 = atomicExchange((&workgroup_struct.atomic_arr[1]), 1);
    return;
}
