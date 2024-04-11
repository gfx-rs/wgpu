#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

const uint SIZE = 128u;

shared int arr_i32_[128];


void main() {
    arr_i32_[gl_LocalInvocationIndex] = 0;
    arr_i32_[gl_LocalInvocationIndex + 4u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 8u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 12u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 16u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 20u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 24u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 28u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 32u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 36u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 40u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 44u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 48u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 52u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 56u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 60u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 64u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 68u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 72u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 76u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 80u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 84u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 88u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 92u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 96u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 100u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 104u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 108u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 112u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 116u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 120u] = 0;
    arr_i32_[gl_LocalInvocationIndex + 124u] = 0;
    memoryBarrierShared();
    barrier();
    uvec3 workgroup_id = gl_WorkGroupID;
    memoryBarrierShared();
    barrier();
    int _e4 = arr_i32_[workgroup_id.x];
    memoryBarrierShared();
    barrier();
    if ((_e4 > 10)) {
        memoryBarrierShared();
        barrier();
        return;
    } else {
        return;
    }
}

