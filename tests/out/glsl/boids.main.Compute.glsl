#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 pos;
    vec2 vel;
};
struct SimParams {
    float deltaT;
    float rule1Distance;
    float rule2Distance;
    float rule3Distance;
    float rule1Scale;
    float rule2Scale;
    float rule3Scale;
};
uniform SimParams_block_0Compute { SimParams _group_0_binding_0; };

layout(std430) readonly buffer Particles_block_1Compute {
    Particle particles[];
} _group_0_binding_1;

layout(std430) buffer Particles_block_2Compute {
    Particle particles[];
} _group_0_binding_2;


void main() {
    uvec3 global_invocation_id = gl_GlobalInvocationID;
    vec2 vPos = vec2(0.0, 0.0);
    vec2 vVel = vec2(0.0, 0.0);
    vec2 cMass = vec2(0.0, 0.0);
    vec2 cVel = vec2(0.0, 0.0);
    vec2 colVel = vec2(0.0, 0.0);
    int cMassCount = 0;
    int cVelCount = 0;
    vec2 pos = vec2(0.0, 0.0);
    vec2 vel = vec2(0.0, 0.0);
    uint i = 0u;
    uint index = global_invocation_id.x;
    if ((index >= 1500u)) {
        return;
    }
    vec2 _e10 = _group_0_binding_1.particles[index].pos;
    vPos = _e10;
    vec2 _e15 = _group_0_binding_1.particles[index].vel;
    vVel = _e15;
    cMass = vec2(0.0, 0.0);
    cVel = vec2(0.0, 0.0);
    colVel = vec2(0.0, 0.0);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
        uint _e86 = i;
        i = (_e86 + 1u);
        }
        loop_init = false;
        uint _e37 = i;
        if ((_e37 >= 1500u)) {
            break;
        }
        uint _e39 = i;
        if ((_e39 == index)) {
            continue;
        }
        uint _e42 = i;
        vec2 _e45 = _group_0_binding_1.particles[_e42].pos;
        pos = _e45;
        uint _e47 = i;
        vec2 _e50 = _group_0_binding_1.particles[_e47].vel;
        vel = _e50;
        vec2 _e51 = pos;
        vec2 _e52 = vPos;
        float _e55 = _group_0_binding_0.rule1Distance;
        if ((distance(_e51, _e52) < _e55)) {
            vec2 _e57 = cMass;
            vec2 _e58 = pos;
            cMass = (_e57 + _e58);
            int _e60 = cMassCount;
            cMassCount = (_e60 + 1);
        }
        vec2 _e63 = pos;
        vec2 _e64 = vPos;
        float _e67 = _group_0_binding_0.rule2Distance;
        if ((distance(_e63, _e64) < _e67)) {
            vec2 _e69 = colVel;
            vec2 _e70 = pos;
            vec2 _e71 = vPos;
            colVel = (_e69 - (_e70 - _e71));
        }
        vec2 _e74 = pos;
        vec2 _e75 = vPos;
        float _e78 = _group_0_binding_0.rule3Distance;
        if ((distance(_e74, _e75) < _e78)) {
            vec2 _e80 = cVel;
            vec2 _e81 = vel;
            cVel = (_e80 + _e81);
            int _e83 = cVelCount;
            cVelCount = (_e83 + 1);
        }
    }
    int _e89 = cMassCount;
    if ((_e89 > 0)) {
        vec2 _e92 = cMass;
        int _e93 = cMassCount;
        vec2 _e97 = vPos;
        cMass = ((_e92 / vec2(float(_e93))) - _e97);
    }
    int _e99 = cVelCount;
    if ((_e99 > 0)) {
        vec2 _e102 = cVel;
        int _e103 = cVelCount;
        cVel = (_e102 / vec2(float(_e103)));
    }
    vec2 _e107 = vVel;
    vec2 _e108 = cMass;
    float _e110 = _group_0_binding_0.rule1Scale;
    vec2 _e113 = colVel;
    float _e115 = _group_0_binding_0.rule2Scale;
    vec2 _e118 = cVel;
    float _e120 = _group_0_binding_0.rule3Scale;
    vVel = (((_e107 + (_e108 * _e110)) + (_e113 * _e115)) + (_e118 * _e120));
    vec2 _e123 = vVel;
    vec2 _e125 = vVel;
    vVel = (normalize(_e123) * clamp(length(_e125), 0.0, 0.10000000149011612));
    vec2 _e131 = vPos;
    vec2 _e132 = vVel;
    float _e134 = _group_0_binding_0.deltaT;
    vPos = (_e131 + (_e132 * _e134));
    float _e138 = vPos.x;
    if ((_e138 < -1.0)) {
        vPos.x = 1.0;
    }
    float _e144 = vPos.x;
    if ((_e144 > 1.0)) {
        vPos.x = -1.0;
    }
    float _e150 = vPos.y;
    if ((_e150 < -1.0)) {
        vPos.y = 1.0;
    }
    float _e156 = vPos.y;
    if ((_e156 > 1.0)) {
        vPos.y = -1.0;
    }
    vec2 _e164 = vPos;
    _group_0_binding_2.particles[index].pos = _e164;
    vec2 _e168 = vVel;
    _group_0_binding_2.particles[index].vel = _e168;
    return;
}

