#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

const uint SIZE = 128u;

shared int arr_i32_[128];


void main() {
    uvec3 workgroup_id = gl_WorkGroupID;
    uint local_invocation_index = gl_LocalInvocationIndex;
    uint zero_init_index = 0u;
    zero_init_index = local_invocation_index;
    while(true) {
        uint _e11 = zero_init_index;
        if ((_e11 >= 128u)) {
            break;
        }
        arr_i32_[_e11] = 0;
        zero_init_index = (_e11 + 4u);
    }
    memoryBarrierShared();
    barrier();
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

