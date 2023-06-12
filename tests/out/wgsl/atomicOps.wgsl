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
    let _e59 = atomicAdd((&storage_atomic_scalar), 1u);
    let _e64 = atomicAdd((&storage_atomic_arr[1]), 1);
    let _e68 = atomicAdd((&storage_struct.atomic_scalar), 1u);
    let _e74 = atomicAdd((&storage_struct.atomic_arr[1]), 1);
    let _e77 = atomicAdd((&workgroup_atomic_scalar), 1u);
    let _e82 = atomicAdd((&workgroup_atomic_arr[1]), 1);
    let _e86 = atomicAdd((&workgroup_struct.atomic_scalar), 1u);
    let _e92 = atomicAdd((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e95 = atomicSub((&storage_atomic_scalar), 1u);
    let _e100 = atomicSub((&storage_atomic_arr[1]), 1);
    let _e104 = atomicSub((&storage_struct.atomic_scalar), 1u);
    let _e110 = atomicSub((&storage_struct.atomic_arr[1]), 1);
    let _e113 = atomicSub((&workgroup_atomic_scalar), 1u);
    let _e118 = atomicSub((&workgroup_atomic_arr[1]), 1);
    let _e122 = atomicSub((&workgroup_struct.atomic_scalar), 1u);
    let _e128 = atomicSub((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e131 = atomicMax((&storage_atomic_scalar), 1u);
    let _e136 = atomicMax((&storage_atomic_arr[1]), 1);
    let _e140 = atomicMax((&storage_struct.atomic_scalar), 1u);
    let _e146 = atomicMax((&storage_struct.atomic_arr[1]), 1);
    let _e149 = atomicMax((&workgroup_atomic_scalar), 1u);
    let _e154 = atomicMax((&workgroup_atomic_arr[1]), 1);
    let _e158 = atomicMax((&workgroup_struct.atomic_scalar), 1u);
    let _e164 = atomicMax((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e167 = atomicMin((&storage_atomic_scalar), 1u);
    let _e172 = atomicMin((&storage_atomic_arr[1]), 1);
    let _e176 = atomicMin((&storage_struct.atomic_scalar), 1u);
    let _e182 = atomicMin((&storage_struct.atomic_arr[1]), 1);
    let _e185 = atomicMin((&workgroup_atomic_scalar), 1u);
    let _e190 = atomicMin((&workgroup_atomic_arr[1]), 1);
    let _e194 = atomicMin((&workgroup_struct.atomic_scalar), 1u);
    let _e200 = atomicMin((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e203 = atomicAnd((&storage_atomic_scalar), 1u);
    let _e208 = atomicAnd((&storage_atomic_arr[1]), 1);
    let _e212 = atomicAnd((&storage_struct.atomic_scalar), 1u);
    let _e218 = atomicAnd((&storage_struct.atomic_arr[1]), 1);
    let _e221 = atomicAnd((&workgroup_atomic_scalar), 1u);
    let _e226 = atomicAnd((&workgroup_atomic_arr[1]), 1);
    let _e230 = atomicAnd((&workgroup_struct.atomic_scalar), 1u);
    let _e236 = atomicAnd((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e239 = atomicOr((&storage_atomic_scalar), 1u);
    let _e244 = atomicOr((&storage_atomic_arr[1]), 1);
    let _e248 = atomicOr((&storage_struct.atomic_scalar), 1u);
    let _e254 = atomicOr((&storage_struct.atomic_arr[1]), 1);
    let _e257 = atomicOr((&workgroup_atomic_scalar), 1u);
    let _e262 = atomicOr((&workgroup_atomic_arr[1]), 1);
    let _e266 = atomicOr((&workgroup_struct.atomic_scalar), 1u);
    let _e272 = atomicOr((&workgroup_struct.atomic_arr[1]), 1);
    workgroupBarrier();
    let _e275 = atomicXor((&storage_atomic_scalar), 1u);
    let _e280 = atomicXor((&storage_atomic_arr[1]), 1);
    let _e284 = atomicXor((&storage_struct.atomic_scalar), 1u);
    let _e290 = atomicXor((&storage_struct.atomic_arr[1]), 1);
    let _e293 = atomicXor((&workgroup_atomic_scalar), 1u);
    let _e298 = atomicXor((&workgroup_atomic_arr[1]), 1);
    let _e302 = atomicXor((&workgroup_struct.atomic_scalar), 1u);
    let _e308 = atomicXor((&workgroup_struct.atomic_arr[1]), 1);
    let _e311 = atomicExchange((&storage_atomic_scalar), 1u);
    let _e316 = atomicExchange((&storage_atomic_arr[1]), 1);
    let _e320 = atomicExchange((&storage_struct.atomic_scalar), 1u);
    let _e326 = atomicExchange((&storage_struct.atomic_arr[1]), 1);
    let _e329 = atomicExchange((&workgroup_atomic_scalar), 1u);
    let _e334 = atomicExchange((&workgroup_atomic_arr[1]), 1);
    let _e338 = atomicExchange((&workgroup_struct.atomic_scalar), 1u);
    let _e344 = atomicExchange((&workgroup_struct.atomic_arr[1]), 1);
    return;
}
