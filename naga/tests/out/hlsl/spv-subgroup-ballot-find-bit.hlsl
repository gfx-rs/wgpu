struct type_1 {
    uint member;
};

static uint global = (uint)0;
RWByteAddressBuffer global_1 : register(u0);

struct ComputeInput_main {
};

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint param = WaveGetLaneIndex();
    global = param;
    uint _e5 = global;
    const uint4 _e7 = WaveActiveBallot((_e5 != 0u));
    const uint unnamed = _e7.x;
    const uint unnamed_1 = _e7.y;
    const uint unnamed_2 = _e7.z;
    const uint unnamed_3 = _e7.w;
    const uint _e8 = (unnamed != 0u ? firstbitlow(unnamed) : (unnamed_1 != 0u ? firstbitlow(unnamed_1) + 32u : (unnamed_2 != 0u ? firstbitlow(unnamed_2) + 64u : firstbitlow(unnamed_3) + 96u)));
    const uint unnamed_4 = _e7.x;
    const uint unnamed_5 = _e7.y;
    const uint unnamed_6 = _e7.z;
    const uint unnamed_7 = _e7.w;
    const uint _e9 = (unnamed_7 != 0u ? firstbithigh(unnamed_7) + 96u : (unnamed_6 != 0u ? firstbithigh(unnamed_6) + 64u : (unnamed_5 != 0u ? firstbithigh(unnamed_5) + 32u : firstbithigh(unnamed_4))));
    global_1.Store(0, asuint((_e8 + _e9)));
}
