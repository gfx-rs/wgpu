#version 310 es

precision highp float;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 pos;
    vec2 vel;
};

uniform SimParams_block_0 {
    float deltaT;
    float rule1Distance;
    float rule2Distance;
    float rule3Distance;
    float rule1Scale;
    float rule2Scale;
    float rule3Scale;
} _group_0_binding_0;
readonly buffer Particles_block_1 {
    Particle particles[];
} _group_0_binding_1;
buffer Particles_block_2 {
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
    vec2 pos1;
    vec2 vel1;
    uint i = 0u;
    if((global_invocation_id.x >= 1500u)) {
        return;
    }
    vPos = _group_0_binding_1.particles[global_invocation_id.x].pos;
    vVel = _group_0_binding_1.particles[global_invocation_id.x].vel;
    cMass = vec2(0.0, 0.0);
    cVel = vec2(0.0, 0.0);
    colVel = vec2(0.0, 0.0);
    while(true) {
        if((i >= 1500u)) {
            break;
        }
        if((i == global_invocation_id.x)) {
            continue;
        }
        pos1 = _group_0_binding_1.particles[i].pos;
        vel1 = _group_0_binding_1.particles[i].vel;
        if((distance(pos1, vPos) < _group_0_binding_0.rule1Distance)) {
            cMass = (cMass + pos1);
            cMassCount = (cMassCount + 1);
        }
        if((distance(pos1, vPos) < _group_0_binding_0.rule2Distance)) {
            colVel = (colVel - (pos1 - vPos));
        }
        if((distance(pos1, vPos) < _group_0_binding_0.rule3Distance)) {
            cVel = (cVel + vel1);
            cVelCount = (cVelCount + 1);
        }
        i = (i + 1u);
    }
    if((cMassCount > 0)) {
        cMass = ((cMass / vec2(float(cMassCount))) - vPos);
    }
    if((cVelCount > 0)) {
        cVel = (cVel / vec2(float(cVelCount)));
    }
    vVel = (((vVel + (cMass * _group_0_binding_0.rule1Scale)) + (colVel * _group_0_binding_0.rule2Scale)) + (cVel * _group_0_binding_0.rule3Scale));
    vVel = (normalize(vVel) * clamp(length(vVel), 0.0, 0.1));
    vPos = (vPos + (vVel * _group_0_binding_0.deltaT));
    if((vPos.x < -1.0)) {
        vPos.x = 1.0;
    }
    if((vPos.x > 1.0)) {
        vPos.x = -1.0;
    }
    if((vPos.y < -1.0)) {
        vPos.y = 1.0;
    }
    if((vPos.y > 1.0)) {
        vPos.y = -1.0;
    }
    _group_0_binding_2.particles[global_invocation_id.x].pos = vPos;
    _group_0_binding_2.particles[global_invocation_id.x].vel = vVel;
    return;
}

