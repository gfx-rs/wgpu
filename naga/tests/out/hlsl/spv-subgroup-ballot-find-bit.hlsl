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
    const uint _e6 = (_e5.x != 0u ? firstbitlow(_e5.x) : (_e5.y != 0u ? firstbitlow(_e5.y) + 32u : (_e5.z != 0u ? firstbitlow(_e5.z) + 64u : firstbitlow(_e5.w) + 96u)));
    const uint _e7 = (_e5.w != 0u ? firstbithigh(_e5.w) + 96u : (_e5.z != 0u ? firstbithigh(_e5.z) + 64u : (_e5.y != 0u ? firstbithigh(_e5.y) + 32u : firstbithigh(_e5.x))));
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
