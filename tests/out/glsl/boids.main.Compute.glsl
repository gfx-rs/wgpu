#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 pos;
    vec2 vel;
};

uniform SimParams_block_0Cs {
    float deltaT;
    float rule1Distance;
    float rule2Distance;
    float rule3Distance;
    float rule1Scale;
    float rule2Scale;
    float rule3Scale;
} _group_0_binding_0;

readonly buffer Particles_block_1Cs {
    Particle particles[];
} _group_0_binding_1;

buffer Particles_block_2Cs {
    Particle particles[];
} _group_0_binding_2;


void main() {
    uvec3 global_invocation_id = gl_GlobalInvocationID;
    vec2 vPos;
    vec2 vVel;
    vec2 cMass;
    vec2 cVel;
    vec2 colVel;
    int cMassCount = 0;
    int cVelCount = 0;
    vec2 pos;
    vec2 vel;
    uint i = 0u;
    uint index = global_invocation_id.x;
    if ((index >= 1500u)) {
        return;
    }
    vec2 _expr10 = _group_0_binding_1.particles[index].pos;
    vPos = _expr10;
    vec2 _expr15 = _group_0_binding_1.particles[index].vel;
    vVel = _expr15;
    cMass = vec2(0.0, 0.0);
    cVel = vec2(0.0, 0.0);
    colVel = vec2(0.0, 0.0);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
        uint _expr86 = i;
        i = (_expr86 + 1u);
        }
        loop_init = false;
        uint _expr37 = i;
        if ((_expr37 >= 1500u)) {
            break;
        }
        uint _expr39 = i;
        if ((_expr39 == index)) {
            continue;
        }
        uint _expr42 = i;
        vec2 _expr45 = _group_0_binding_1.particles[_expr42].pos;
        pos = _expr45;
        uint _expr47 = i;
        vec2 _expr50 = _group_0_binding_1.particles[_expr47].vel;
        vel = _expr50;
        vec2 _expr51 = pos;
        vec2 _expr52 = vPos;
        float _expr55 = _group_0_binding_0.rule1Distance;
        if ((distance(_expr51, _expr52) < _expr55)) {
            vec2 _expr57 = cMass;
            vec2 _expr58 = pos;
            cMass = (_expr57 + _expr58);
            int _expr60 = cMassCount;
            cMassCount = (_expr60 + 1);
        }
        vec2 _expr63 = pos;
        vec2 _expr64 = vPos;
        float _expr67 = _group_0_binding_0.rule2Distance;
        if ((distance(_expr63, _expr64) < _expr67)) {
            vec2 _expr69 = colVel;
            vec2 _expr70 = pos;
            vec2 _expr71 = vPos;
            colVel = (_expr69 - (_expr70 - _expr71));
        }
        vec2 _expr74 = pos;
        vec2 _expr75 = vPos;
        float _expr78 = _group_0_binding_0.rule3Distance;
        if ((distance(_expr74, _expr75) < _expr78)) {
            vec2 _expr80 = cVel;
            vec2 _expr81 = vel;
            cVel = (_expr80 + _expr81);
            int _expr83 = cVelCount;
            cVelCount = (_expr83 + 1);
        }
    }
    int _expr89 = cMassCount;
    if ((_expr89 > 0)) {
        vec2 _expr92 = cMass;
        int _expr93 = cMassCount;
        vec2 _expr97 = vPos;
        cMass = ((_expr92 / vec2(float(_expr93))) - _expr97);
    }
    int _expr99 = cVelCount;
    if ((_expr99 > 0)) {
        vec2 _expr102 = cVel;
        int _expr103 = cVelCount;
        cVel = (_expr102 / vec2(float(_expr103)));
    }
    vec2 _expr107 = vVel;
    vec2 _expr108 = cMass;
    float _expr110 = _group_0_binding_0.rule1Scale;
    vec2 _expr113 = colVel;
    float _expr115 = _group_0_binding_0.rule2Scale;
    vec2 _expr118 = cVel;
    float _expr120 = _group_0_binding_0.rule3Scale;
    vVel = (((_expr107 + (_expr108 * _expr110)) + (_expr113 * _expr115)) + (_expr118 * _expr120));
    vec2 _expr123 = vVel;
    vec2 _expr125 = vVel;
    vVel = (normalize(_expr123) * clamp(length(_expr125), 0.0, 0.1));
    vec2 _expr131 = vPos;
    vec2 _expr132 = vVel;
    float _expr134 = _group_0_binding_0.deltaT;
    vPos = (_expr131 + (_expr132 * _expr134));
    vec2 _expr137 = vPos;
    if ((_expr137.x < -1.0)) {
        vPos.x = 1.0;
    }
    vec2 _expr143 = vPos;
    if ((_expr143.x > 1.0)) {
        vPos.x = -1.0;
    }
    vec2 _expr149 = vPos;
    if ((_expr149.y < -1.0)) {
        vPos.y = 1.0;
    }
    vec2 _expr155 = vPos;
    if ((_expr155.y > 1.0)) {
        vPos.y = -1.0;
    }
    vec2 _expr164 = vPos;
    _group_0_binding_2.particles[index].pos = _expr164;
    vec2 _expr168 = vVel;
    _group_0_binding_2.particles[index].vel = _expr168;
    return;
}

