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
    float2 _e8 = asfloat(particlesSrc.Load2(0+index*16+0));
    vPos = _e8;
    float2 _e14 = asfloat(particlesSrc.Load2(8+index*16+0));
    vVel = _e14;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e91 = i;
            i = (_e91 + 1u);
        }
        loop_init = false;
        uint _e36 = i;
        if ((_e36 >= NUM_PARTICLES)) {
            break;
        }
        uint _e39 = i;
        if ((_e39 == index)) {
            continue;
        }
        uint _e43 = i;
        float2 _e46 = asfloat(particlesSrc.Load2(0+_e43*16+0));
        pos = _e46;
        uint _e49 = i;
        float2 _e52 = asfloat(particlesSrc.Load2(8+_e49*16+0));
        vel = _e52;
        float2 _e53 = pos;
        float2 _e54 = vPos;
        float _e58 = params.rule1Distance;
        if ((distance(_e53, _e54) < _e58)) {
            float2 _e60 = cMass;
            float2 _e61 = pos;
            cMass = (_e60 + _e61);
            int _e63 = cMassCount;
            cMassCount = (_e63 + 1);
        }
        float2 _e66 = pos;
        float2 _e67 = vPos;
        float _e71 = params.rule2Distance;
        if ((distance(_e66, _e67) < _e71)) {
            float2 _e73 = colVel;
            float2 _e74 = pos;
            float2 _e75 = vPos;
            colVel = (_e73 - (_e74 - _e75));
        }
        float2 _e78 = pos;
        float2 _e79 = vPos;
        float _e83 = params.rule3Distance;
        if ((distance(_e78, _e79) < _e83)) {
            float2 _e85 = cVel;
            float2 _e86 = vel;
            cVel = (_e85 + _e86);
            int _e88 = cVelCount;
            cVelCount = (_e88 + 1);
        }
    }
    int _e94 = cMassCount;
    if ((_e94 > 0)) {
        float2 _e97 = cMass;
        int _e98 = cMassCount;
        float2 _e102 = vPos;
        cMass = ((_e97 / (float(_e98)).xx) - _e102);
    }
    int _e104 = cVelCount;
    if ((_e104 > 0)) {
        float2 _e107 = cVel;
        int _e108 = cVelCount;
        cVel = (_e107 / (float(_e108)).xx);
    }
    float2 _e112 = vVel;
    float2 _e113 = cMass;
    float _e116 = params.rule1Scale;
    float2 _e119 = colVel;
    float _e122 = params.rule2Scale;
    float2 _e125 = cVel;
    float _e128 = params.rule3Scale;
    vVel = (((_e112 + (_e113 * _e116)) + (_e119 * _e122)) + (_e125 * _e128));
    float2 _e131 = vVel;
    float2 _e133 = vVel;
    vVel = (normalize(_e131) * clamp(length(_e133), 0.0, 0.1));
    float2 _e139 = vPos;
    float2 _e140 = vVel;
    float _e143 = params.deltaT;
    vPos = (_e139 + (_e140 * _e143));
    float _e147 = vPos.x;
    if ((_e147 < -1.0)) {
        vPos.x = 1.0;
    }
    float _e153 = vPos.x;
    if ((_e153 > 1.0)) {
        vPos.x = -1.0;
    }
    float _e159 = vPos.y;
    if ((_e159 < -1.0)) {
        vPos.y = 1.0;
    }
    float _e165 = vPos.y;
    if ((_e165 > 1.0)) {
        vPos.y = -1.0;
    }
    float2 _e174 = vPos;
    particlesDst.Store2(0+index*16+0, asuint(_e174));
    float2 _e179 = vVel;
    particlesDst.Store2(8+index*16+0, asuint(_e179));
    return;
}
