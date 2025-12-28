#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

shared uint wg_scalar;

shared int wg_signed;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        wg_scalar = 0u;
        wg_signed = 0;
    }
    memoryBarrierShared();
    barrier();
    uvec3 workgroup_id = gl_WorkGroupID;
    uvec3 local_id = gl_LocalInvocationID;
    bool local = false;
    uint active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    uint _e11 = atomicOr(wg_scalar, uint((active_tile_index >= 64u)));
    int _e14 = atomicAdd(wg_signed, 1);
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    uint _e16 = wg_scalar;
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    int _e18 = wg_signed;
    memoryBarrierShared();
    barrier();
    if ((_e16 == 0u)) {
        local = (_e18 > 0);
    } else {
        local = false;
    }
    bool _e26 = local;
    if (_e26) {
        return;
    } else {
        return;
    }
}

