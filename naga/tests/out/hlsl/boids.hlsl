struct Particle {
    float2 pos;
    float2 vel;
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

static const uint NUM_PARTICLES = 1500u;

cbuffer params : register(b0) { SimParams params; }
ByteAddressBuffer particlesSrc : register(t1);
RWByteAddressBuffer particlesDst : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 global_invocation_id : SV_DispatchThreadID)
{
    float2 vPos = (float2)0;
    float2 vVel = (float2)0;
    float2 cMass = float2(0.0, 0.0);
    float2 cVel = float2(0.0, 0.0);
    float2 colVel = float2(0.0, 0.0);
    int cMassCount = 0;
    int cVelCount = 0;
    float2 pos = (float2)0;
    float2 vel = (float2)0;
    uint i = 0u;

    uint index = global_invocation_id.x;
    if ((index >= NUM_PARTICLES)) {
        return;
    }
    float2 _expr8 = asfloat(particlesSrc.Load2(0+index*16+0));
    vPos = _expr8;
    float2 _expr14 = asfloat(particlesSrc.Load2(8+index*16+0));
    vVel = _expr14;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _expr91 = i;
            i = (_expr91 + 1u);
        }
        loop_init = false;
        uint _expr36 = i;
        if ((_expr36 >= NUM_PARTICLES)) {
            break;
        }
        uint _expr39 = i;
        if ((_expr39 == index)) {
            continue;
        }
        uint _expr43 = i;
        float2 _expr46 = asfloat(particlesSrc.Load2(0+_expr43*16+0));
        pos = _expr46;
        uint _expr49 = i;
        float2 _expr52 = asfloat(particlesSrc.Load2(8+_expr49*16+0));
        vel = _expr52;
        float2 _expr53 = pos;
        float2 _expr54 = vPos;
        float _expr58 = params.rule1Distance;
        if ((distance(_expr53, _expr54) < _expr58)) {
            float2 _expr60 = cMass;
            float2 _expr61 = pos;
            cMass = (_expr60 + _expr61);
            int _expr63 = cMassCount;
            cMassCount = (_expr63 + 1);
        }
        float2 _expr66 = pos;
        float2 _expr67 = vPos;
        float _expr71 = params.rule2Distance;
        if ((distance(_expr66, _expr67) < _expr71)) {
            float2 _expr73 = colVel;
            float2 _expr74 = pos;
            float2 _expr75 = vPos;
            colVel = (_expr73 - (_expr74 - _expr75));
        }
        float2 _expr78 = pos;
        float2 _expr79 = vPos;
        float _expr83 = params.rule3Distance;
        if ((distance(_expr78, _expr79) < _expr83)) {
            float2 _expr85 = cVel;
            float2 _expr86 = vel;
            cVel = (_expr85 + _expr86);
            int _expr88 = cVelCount;
            cVelCount = (_expr88 + 1);
        }
    }
    int _expr94 = cMassCount;
    if ((_expr94 > 0)) {
        float2 _expr97 = cMass;
        int _expr98 = cMassCount;
        float2 _expr102 = vPos;
        cMass = ((_expr97 / (float(_expr98)).xx) - _expr102);
    }
    int _expr104 = cVelCount;
    if ((_expr104 > 0)) {
        float2 _expr107 = cVel;
        int _expr108 = cVelCount;
        cVel = (_expr107 / (float(_expr108)).xx);
    }
    float2 _expr112 = vVel;
    float2 _expr113 = cMass;
    float _expr116 = params.rule1Scale;
    float2 _expr119 = colVel;
    float _expr122 = params.rule2Scale;
    float2 _expr125 = cVel;
    float _expr128 = params.rule3Scale;
    vVel = (((_expr112 + (_expr113 * _expr116)) + (_expr119 * _expr122)) + (_expr125 * _expr128));
    float2 _expr131 = vVel;
    float2 _expr133 = vVel;
    vVel = (normalize(_expr131) * clamp(length(_expr133), 0.0, 0.1));
    float2 _expr139 = vPos;
    float2 _expr140 = vVel;
    float _expr143 = params.deltaT;
    vPos = (_expr139 + (_expr140 * _expr143));
    float _expr147 = vPos.x;
    if ((_expr147 < -1.0)) {
        vPos.x = 1.0;
    }
    float _expr153 = vPos.x;
    if ((_expr153 > 1.0)) {
        vPos.x = -1.0;
    }
    float _expr159 = vPos.y;
    if ((_expr159 < -1.0)) {
        vPos.y = 1.0;
    }
    float _expr165 = vPos.y;
    if ((_expr165 > 1.0)) {
        vPos.y = -1.0;
    }
    float2 _expr174 = vPos;
    particlesDst.Store2(0+index*16+0, asuint(_expr174));
    float2 _expr179 = vVel;
    particlesDst.Store2(8+index*16+0, asuint(_expr179));
    return;
}
