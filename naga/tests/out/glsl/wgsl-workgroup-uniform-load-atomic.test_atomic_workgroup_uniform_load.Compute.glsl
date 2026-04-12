#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct AtomicStruct {
    uint atomic_scalar;
    int atomic_arr[2];
};
shared uint wg_scalar;

shared int wg_signed;

shared AtomicStruct wg_struct;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        wg_scalar = 0u;
        wg_signed = 0;
        wg_struct = AtomicStruct(0u, int[2](0, 0));
    }
    memoryBarrierShared();
    barrier();
    uvec3 workgroup_id = gl_WorkGroupID;
    uvec3 local_id = gl_LocalInvocationID;
    bool local = false;
    bool local_1 = false;
    bool local_2 = false;
    uint active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    uint _e11 = atomicOr(wg_scalar, uint((active_tile_index >= 64u)));
    int _e14 = atomicAdd(wg_signed, 1);
    atomicExchange(wg_struct.atomic_scalar, 1u);
    int _e22 = atomicAdd(wg_struct.atomic_arr[0], 1);
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    uint _e24 = wg_scalar;
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    int _e26 = wg_signed;
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    uint _e29 = wg_struct.atomic_scalar;
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    int _e33 = wg_struct.atomic_arr[0];
    memoryBarrierShared();
    barrier();
    if ((_e24 == 0u)) {
        local = (_e26 > 0);
    } else {
        local = false;
    }
    bool _e41 = local;
    if (_e41) {
        local_1 = (_e29 > 0u);
    } else {
        local_1 = false;
    }
    bool _e47 = local_1;
    if (_e47) {
        local_2 = (_e33 > 0);
    } else {
        local_2 = false;
    }
    bool _e53 = local_2;
    if (_e53) {
        return;
    } else {
        return;
    }
}

