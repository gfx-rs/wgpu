#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Padded {
    uvec3 xyz;
    uint tail;
};
const uint W = 64u;
const uint N = 513u;

shared uint flag;

shared uvec2 pair;

shared uint scratch[513];

shared uint counter;

shared uvec3 vectors[3];

shared Padded padded;

layout(std430) buffer type_6_block_0Compute { uint _group_0_binding_0_cs[]; };


uint sum_small() {
    uint sum_1 = 0u;
    uint j_3 = 0u;
    uint _e1 = flag;
    uint _e4 = pair.x;
    uint _e8 = pair.y;
    uint _e13 = padded.xyz.x;
    uint _e18 = padded.xyz.y;
    uint _e23 = padded.xyz.z;
    uint _e27 = padded.tail;
    sum_1 = ((((((_e1 + _e4) + _e8) + _e13) + _e18) + _e23) + _e27);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e55 = j_3;
            j_3 = (_e55 + 1u);
        }
        loop_init = false;
        uint _e32 = j_3;
        if ((_e32 < 3u)) {
        } else {
            break;
        }
        {
            uint _e35 = sum_1;
            uint _e37 = j_3;
            uint _e40 = vectors[_e37].x;
            uint _e42 = j_3;
            uint _e45 = vectors[_e42].y;
            uint _e48 = j_3;
            uint _e51 = vectors[_e48].z;
            sum_1 = (_e35 + ((_e40 + _e45) + _e51));
        }
    }
    uint _e57 = sum_1;
    return _e57;
}

void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        flag = 0u;
        pair = uvec2(0u);
        scratch = uint[513](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        counter = 0u;
        vectors = uvec3[3](uvec3(0u), uvec3(0u), uvec3(0u));
        padded = Padded(uvec3(0u), 0u);
    }
    memoryBarrierShared();
    barrier();
    uint i = gl_LocalInvocationIndex;
    uvec3 group = gl_WorkGroupID;
    uint initial = 0u;
    uint j = 0u;
    uint j_1 = 0u;
    uint sum = 0u;
    uint j_2 = 0u;
    uint base = (group.x * 4u);
    if ((i == 0u)) {
        uint _e7 = sum_small();
        uint _e9 = atomicOr(counter, 0u);
        initial = (_e7 + _e9);
        bool loop_init_1 = true;
        while(true) {
            if (!loop_init_1) {
                uint _e24 = j;
                j = (_e24 + 1u);
            }
            loop_init_1 = false;
            uint _e14 = j;
            if ((_e14 < N)) {
            } else {
                break;
            }
            {
                uint _e17 = initial;
                uint _e19 = j;
                uint _e21 = scratch[_e19];
                initial = (_e17 + _e21);
            }
        }
        uint _e28 = initial;
        _group_0_binding_0_cs[base] = _e28;
    }
    memoryBarrierShared();
    barrier();
    j_1 = i;
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            uint _e39 = j_1;
            j_1 = (_e39 + W);
        }
        loop_init_2 = false;
        uint _e30 = j_1;
        if ((_e30 < N)) {
        } else {
            break;
        }
        {
            uint _e34 = j_1;
            uint _e36 = j_1;
            scratch[_e34] = (_e36 + group.x);
        }
    }
    uint _e44 = atomicAdd(counter, 1u);
    if ((i == 0u)) {
        flag = 1u;
        pair = uvec2(2u, 3u);
        vectors = uvec3[3](uvec3(4u, 5u, 6u), uvec3(7u, 8u, 9u), uvec3(10u, 11u, 12u));
        padded = Padded(uvec3(13u, 14u, 15u), 16u);
    }
    memoryBarrierShared();
    barrier();
    if ((i == 0u)) {
        bool loop_init_3 = true;
        while(true) {
            if (!loop_init_3) {
                uint _e90 = j_2;
                j_2 = (_e90 + 1u);
            }
            loop_init_3 = false;
            uint _e80 = j_2;
            if ((_e80 < N)) {
            } else {
                break;
            }
            {
                uint _e83 = sum;
                uint _e85 = j_2;
                uint _e87 = scratch[_e85];
                sum = (_e83 + _e87);
            }
        }
        uint _e96 = sum;
        _group_0_binding_0_cs[(base + 1u)] = _e96;
        uint _e102 = atomicOr(counter, 0u);
        _group_0_binding_0_cs[(base + 2u)] = _e102;
        uint _e107 = sum_small();
        _group_0_binding_0_cs[(base + 3u)] = _e107;
        return;
    } else {
        return;
    }
}

