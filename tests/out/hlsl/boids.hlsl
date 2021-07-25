static const uint NUM_PARTICLES = 1500;

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

cbuffer params : register(b0) { SimParams params; }
ByteAddressBuffer particlesSrc : register(t1);
RWByteAddressBuffer particlesDst : register(u2);

struct ComputeInput_main {
    uint3 global_invocation_id1 : SV_DispatchThreadID;
};

[numthreads(64, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    float2 vPos = (float2)0;
    float2 vVel = (float2)0;
    float2 cMass = (float2)0;
    float2 cVel = (float2)0;
    float2 colVel = (float2)0;
    int cMassCount = 0;
    int cVelCount = 0;
    float2 pos = (float2)0;
    float2 vel = (float2)0;
    uint i = 0u;

    uint index = computeinput_main.global_invocation_id1.x;
    if ((index >= NUM_PARTICLES)) {
        return;
    }
    float2 _expr10 = asfloat(particlesSrc.Load2(0+index*4+0));
    vPos = _expr10;
    float2 _expr15 = asfloat(particlesSrc.Load2(4+index*4+0));
    vVel = _expr15;
    cMass = float2(0.0, 0.0);
    cVel = float2(0.0, 0.0);
    colVel = float2(0.0, 0.0);
    while(true) {
        uint _expr37 = i;
        if ((_expr37 >= NUM_PARTICLES)) {
            break;
        }
        uint _expr39 = i;
        if ((_expr39 == index)) {
            continue;
        }
        uint _expr42 = i;
        float2 _expr45 = asfloat(particlesSrc.Load2(0+_expr42*4+0));
        pos = _expr45;
        uint _expr47 = i;
        float2 _expr50 = asfloat(particlesSrc.Load2(4+_expr47*4+0));
        vel = _expr50;
        float2 _expr51 = pos;
        float2 _expr52 = vPos;
        float _expr55 = params.rule1Distance;
        if ((distance(_expr51, _expr52) < _expr55)) {
            float2 _expr57 = cMass;
            float2 _expr58 = pos;
            cMass = (_expr57 + _expr58);
            int _expr60 = cMassCount;
            cMassCount = (_expr60 + 1);
        }
        float2 _expr63 = pos;
        float2 _expr64 = vPos;
        float _expr67 = params.rule2Distance;
        if ((distance(_expr63, _expr64) < _expr67)) {
            float2 _expr69 = colVel;
            float2 _expr70 = pos;
            float2 _expr71 = vPos;
            colVel = (_expr69 - (_expr70 - _expr71));
        }
        float2 _expr74 = pos;
        float2 _expr75 = vPos;
        float _expr78 = params.rule3Distance;
        if ((distance(_expr74, _expr75) < _expr78)) {
            float2 _expr80 = cVel;
            float2 _expr81 = vel;
            cVel = (_expr80 + _expr81);
            int _expr83 = cVelCount;
            cVelCount = (_expr83 + 1);
        }
        uint _expr86 = i;
        i = (_expr86 + 1u);
    }
    int _expr89 = cMassCount;
    if ((_expr89 > 0)) {
        float2 _expr92 = cMass;
        int _expr93 = cMassCount;
        float2 _expr97 = vPos;
        cMass = ((_expr92 / float2(float(_expr93).xx)) - _expr97);
    }
    int _expr99 = cVelCount;
    if ((_expr99 > 0)) {
        float2 _expr102 = cVel;
        int _expr103 = cVelCount;
        cVel = (_expr102 / float2(float(_expr103).xx));
    }
    float2 _expr107 = vVel;
    float2 _expr108 = cMass;
    float _expr110 = params.rule1Scale;
    float2 _expr113 = colVel;
    float _expr115 = params.rule2Scale;
    float2 _expr118 = cVel;
    float _expr120 = params.rule3Scale;
    vVel = (((_expr107 + (_expr108 * _expr110)) + (_expr113 * _expr115)) + (_expr118 * _expr120));
    float2 _expr123 = vVel;
    float2 _expr125 = vVel;
    vVel = (normalize(_expr123) * clamp(length(_expr125), 0.0, 0.1));
    float2 _expr131 = vPos;
    float2 _expr132 = vVel;
    float _expr134 = params.deltaT;
    vPos = (_expr131 + (_expr132 * _expr134));
    float2 _expr137 = vPos;
    if ((_expr137.x < -1.0)) {
        vPos.x = 1.0;
    }
    float2 _expr143 = vPos;
    if ((_expr143.x > 1.0)) {
        vPos.x = -1.0;
    }
    float2 _expr149 = vPos;
    if ((_expr149.y < -1.0)) {
        vPos.y = 1.0;
    }
    float2 _expr155 = vPos;
    if ((_expr155.y > 1.0)) {
        vPos.y = -1.0;
    }
    float2 _expr164 = vPos;
    particlesDst.Store2(0+index*4+0, asuint(_expr164));
    float2 _expr168 = vVel;
    particlesDst.Store2(4+index*4+0, asuint(_expr168));
    return;
}
