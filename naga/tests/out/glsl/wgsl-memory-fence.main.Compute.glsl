#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer Data_block_0Compute {
    uint values[];
} _group_0_binding_0_cs;

layout(std430) buffer type_2_block_1Compute { uint _group_0_binding_1_cs; };

shared uint stage;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        stage = 0u;
    }
    memoryBarrierShared();
    barrier();
    uint index = gl_LocalInvocationIndex;
    uint spins = 0u;
    _group_0_binding_0_cs.values[index] = index;
    memoryBarrierBuffer();
    if ((index == 0u)) {
        atomicExchange(_group_0_binding_1_cs, 1u);
    }
    while(true) {
        uint _e11 = atomicOr(_group_0_binding_1_cs, 0u);
        if ((_e11 != 0u)) {
            memoryBarrierBuffer();
            uint _e18 = _group_0_binding_0_cs.values[0];
            stage = _e18;
            memoryBarrierShared();
            break;
        }
        uint _e19 = spins;
        spins = (_e19 + 1u);
        uint _e22 = spins;
        if ((_e22 > 65536u)) {
            break;
        }
    }
    return;
}
