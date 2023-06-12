#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct Struct {
    uint atomic_scalar;
    int atomic_arr[2];
};
layout(std430) buffer type_block_0Compute { uint _group_0_binding_0_cs; };

layout(std430) buffer type_2_block_1Compute { int _group_0_binding_1_cs[2]; };

layout(std430) buffer Struct_block_2Compute { Struct _group_0_binding_2_cs; };

shared uint workgroup_atomic_scalar;

shared int workgroup_atomic_arr[2];

shared Struct workgroup_struct;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        workgroup_atomic_scalar = 0u;
        workgroup_atomic_arr = int[2](0, 0);
        workgroup_struct = Struct(0u, int[2](0, 0));
    }
    memoryBarrierShared();
    barrier();
    uvec3 id = gl_LocalInvocationID;
    _group_0_binding_0_cs = 1u;
    _group_0_binding_1_cs[1] = 1;
    _group_0_binding_2_cs.atomic_scalar = 1u;
    _group_0_binding_2_cs.atomic_arr[1] = 1;
    workgroup_atomic_scalar = 1u;
    workgroup_atomic_arr[1] = 1;
    workgroup_struct.atomic_scalar = 1u;
    workgroup_struct.atomic_arr[1] = 1;
    memoryBarrierShared();
    barrier();
    uint l0_ = _group_0_binding_0_cs;
    int l1_ = _group_0_binding_1_cs[1];
    uint l2_ = _group_0_binding_2_cs.atomic_scalar;
    int l3_ = _group_0_binding_2_cs.atomic_arr[1];
    uint l4_ = workgroup_atomic_scalar;
    int l5_ = workgroup_atomic_arr[1];
    uint l6_ = workgroup_struct.atomic_scalar;
    int l7_ = workgroup_struct.atomic_arr[1];
    memoryBarrierShared();
    barrier();
    uint _e59 = atomicAdd(_group_0_binding_0_cs, 1u);
    int _e64 = atomicAdd(_group_0_binding_1_cs[1], 1);
    uint _e68 = atomicAdd(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e74 = atomicAdd(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e77 = atomicAdd(workgroup_atomic_scalar, 1u);
    int _e82 = atomicAdd(workgroup_atomic_arr[1], 1);
    uint _e86 = atomicAdd(workgroup_struct.atomic_scalar, 1u);
    int _e92 = atomicAdd(workgroup_struct.atomic_arr[1], 1);
    memoryBarrierShared();
    barrier();
    uint _e95 = atomicAdd(_group_0_binding_0_cs, -1u);
    int _e100 = atomicAdd(_group_0_binding_1_cs[1], -1);
    uint _e104 = atomicAdd(_group_0_binding_2_cs.atomic_scalar, -1u);
    int _e110 = atomicAdd(_group_0_binding_2_cs.atomic_arr[1], -1);
    uint _e113 = atomicAdd(workgroup_atomic_scalar, -1u);
    int _e118 = atomicAdd(workgroup_atomic_arr[1], -1);
    uint _e122 = atomicAdd(workgroup_struct.atomic_scalar, -1u);
    int _e128 = atomicAdd(workgroup_struct.atomic_arr[1], -1);
    memoryBarrierShared();
    barrier();
    uint _e131 = atomicMax(_group_0_binding_0_cs, 1u);
    int _e136 = atomicMax(_group_0_binding_1_cs[1], 1);
    uint _e140 = atomicMax(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e146 = atomicMax(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e149 = atomicMax(workgroup_atomic_scalar, 1u);
    int _e154 = atomicMax(workgroup_atomic_arr[1], 1);
    uint _e158 = atomicMax(workgroup_struct.atomic_scalar, 1u);
    int _e164 = atomicMax(workgroup_struct.atomic_arr[1], 1);
    memoryBarrierShared();
    barrier();
    uint _e167 = atomicMin(_group_0_binding_0_cs, 1u);
    int _e172 = atomicMin(_group_0_binding_1_cs[1], 1);
    uint _e176 = atomicMin(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e182 = atomicMin(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e185 = atomicMin(workgroup_atomic_scalar, 1u);
    int _e190 = atomicMin(workgroup_atomic_arr[1], 1);
    uint _e194 = atomicMin(workgroup_struct.atomic_scalar, 1u);
    int _e200 = atomicMin(workgroup_struct.atomic_arr[1], 1);
    memoryBarrierShared();
    barrier();
    uint _e203 = atomicAnd(_group_0_binding_0_cs, 1u);
    int _e208 = atomicAnd(_group_0_binding_1_cs[1], 1);
    uint _e212 = atomicAnd(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e218 = atomicAnd(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e221 = atomicAnd(workgroup_atomic_scalar, 1u);
    int _e226 = atomicAnd(workgroup_atomic_arr[1], 1);
    uint _e230 = atomicAnd(workgroup_struct.atomic_scalar, 1u);
    int _e236 = atomicAnd(workgroup_struct.atomic_arr[1], 1);
    memoryBarrierShared();
    barrier();
    uint _e239 = atomicOr(_group_0_binding_0_cs, 1u);
    int _e244 = atomicOr(_group_0_binding_1_cs[1], 1);
    uint _e248 = atomicOr(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e254 = atomicOr(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e257 = atomicOr(workgroup_atomic_scalar, 1u);
    int _e262 = atomicOr(workgroup_atomic_arr[1], 1);
    uint _e266 = atomicOr(workgroup_struct.atomic_scalar, 1u);
    int _e272 = atomicOr(workgroup_struct.atomic_arr[1], 1);
    memoryBarrierShared();
    barrier();
    uint _e275 = atomicXor(_group_0_binding_0_cs, 1u);
    int _e280 = atomicXor(_group_0_binding_1_cs[1], 1);
    uint _e284 = atomicXor(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e290 = atomicXor(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e293 = atomicXor(workgroup_atomic_scalar, 1u);
    int _e298 = atomicXor(workgroup_atomic_arr[1], 1);
    uint _e302 = atomicXor(workgroup_struct.atomic_scalar, 1u);
    int _e308 = atomicXor(workgroup_struct.atomic_arr[1], 1);
    uint _e311 = atomicExchange(_group_0_binding_0_cs, 1u);
    int _e316 = atomicExchange(_group_0_binding_1_cs[1], 1);
    uint _e320 = atomicExchange(_group_0_binding_2_cs.atomic_scalar, 1u);
    int _e326 = atomicExchange(_group_0_binding_2_cs.atomic_arr[1], 1);
    uint _e329 = atomicExchange(workgroup_atomic_scalar, 1u);
    int _e334 = atomicExchange(workgroup_atomic_arr[1], 1);
    uint _e338 = atomicExchange(workgroup_struct.atomic_scalar, 1u);
    int _e344 = atomicExchange(workgroup_struct.atomic_arr[1], 1);
    return;
}

