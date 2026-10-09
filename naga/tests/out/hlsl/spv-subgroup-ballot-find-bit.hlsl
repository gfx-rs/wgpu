struct type_1 {
    uint member;
};

static uint global = (uint)0;
RWByteAddressBuffer global_1 : register(u0);

struct ComputeInput_main {
};

void function()
{
    uint _e3 = global;
    const uint4 _e5 = WaveActiveBallot((_e3 != 0u));
    const uint unnamed = _e5.x;
    const uint unnamed_1 = _e5.y;
    const uint unnamed_2 = _e5.z;
    const uint unnamed_3 = _e5.w;
    const uint _e6 = (unnamed != 0u ? firstbitlow(unnamed) : (unnamed_1 != 0u ? firstbitlow(unnamed_1) + 32u : (unnamed_2 != 0u ? firstbitlow(unnamed_2) + 64u : firstbitlow(unnamed_3) + 96u)));
    const uint unnamed_4 = _e5.x;
    const uint unnamed_5 = _e5.y;
    const uint unnamed_6 = _e5.z;
    const uint unnamed_7 = _e5.w;
    const uint _e7 = (unnamed_7 != 0u ? firstbithigh(unnamed_7) + 96u : (unnamed_6 != 0u ? firstbithigh(unnamed_6) + 64u : (unnamed_5 != 0u ? firstbithigh(unnamed_5) + 32u : firstbithigh(unnamed_4))));
    global_1.Store(0, asuint((_e6 + _e7)));
    return;
}

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint param = WaveGetLaneIndex();
    global = param;
    function();
}
