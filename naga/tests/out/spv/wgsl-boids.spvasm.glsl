#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle
{
    vec2 pos;
    vec2 vel;
};

struct SimParams
{
    float deltaT;
    float rule1Distance;
    float rule2Distance;
    float rule3Distance;
    float rule1Scale;
    float rule2Scale;
    float rule3Scale;
};

layout(set = 0, binding = 0, std140) uniform params
{
    SimParams _m0;
} params_1;

layout(set = 0, binding = 1, std430) readonly buffer Particles
{
    Particle particles[];
} particlesSrc;

layout(set = 0, binding = 2, std430) buffer particlesDst
{
    Particle particles[];
} particlesDst_1;

void main()
{
    uint i = 0u;
    int cVelCount = 0;
    vec2 cVel = vec2(0.0);
    vec2 vPos = vec2(0.0);
    vec2 pos = vec2(0.0);
    vec2 colVel = vec2(0.0);
    vec2 vVel = vec2(0.0);
    vec2 vel = vec2(0.0);
    int cMassCount = 0;
    vec2 cMass = vec2(0.0);
    uvec2 loop_bound = uvec2(4294967295u);
    if (gl_GlobalInvocationID.x >= 1500u)
    {
        return;
    }
    vPos = particlesSrc.particles[gl_GlobalInvocationID.x].pos;
    vVel = particlesSrc.particles[gl_GlobalInvocationID.x].vel;
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (i >= 1500u)
        {
            break;
        }
        if (i == gl_GlobalInvocationID.x)
        {
            uint _144 = i;
            i = _144 + 1u;
            continue;
        }
        pos = particlesSrc.particles[i].pos;
        vel = particlesSrc.particles[i].vel;
        if (distance(pos, vPos) < params_1._m0.rule1Distance)
        {
            cMass += pos;
            cMassCount++;
        }
        if (distance(pos, vPos) < params_1._m0.rule2Distance)
        {
            colVel -= (pos - vPos);
        }
        if (distance(pos, vPos) < params_1._m0.rule3Distance)
        {
            cVel += vel;
            cVelCount++;
        }
        uint _144 = i;
        i = _144 + 1u;
        continue;
    }
    if (cMassCount > 0)
    {
        cMass = (cMass / vec2(float(cMassCount))) - vPos;
    }
    if (cVelCount > 0)
    {
        cVel /= vec2(float(cVelCount));
    }
    vVel = ((vVel + (cMass * params_1._m0.rule1Scale)) + (colVel * params_1._m0.rule2Scale)) + (cVel * params_1._m0.rule3Scale);
    vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.100000001490116119384765625);
    vPos += (vVel * params_1._m0.deltaT);
    if (vPos.x < (-1.0))
    {
        vPos.x = 1.0;
    }
    if (vPos.x > 1.0)
    {
        vPos.x = -1.0;
    }
    if (vPos.y < (-1.0))
    {
        vPos.y = 1.0;
    }
    if (vPos.y > 1.0)
    {
        vPos.y = -1.0;
    }
    particlesDst_1.particles[gl_GlobalInvocationID.x].pos = vPos;
    particlesDst_1.particles[gl_GlobalInvocationID.x].vel = vVel;
}

