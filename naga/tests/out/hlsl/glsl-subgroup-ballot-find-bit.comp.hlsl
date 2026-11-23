struct Output {
    uint result;
};

RWByteAddressBuffer global : register(u0);
static uint gl_SubgroupInvocationID_1 = (uint)0;
static uint gl_SubgroupSize_1 = (uint)0;

struct ComputeInput_main {
};

uint4 ZeroValueuint4() {
    return (uint4)0;
}

uint ZeroValueuint() {
    return (uint)0;
}

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint gl_SubgroupInvocationID = WaveGetLaneIndex();
    uint gl_SubgroupSize = WaveGetLaneCount();
    uint4 mask = (uint4)0;
    uint lsb = (uint)0;
    uint msb = (uint)0;

    gl_SubgroupInvocationID_1 = gl_SubgroupInvocationID;
    gl_SubgroupSize_1 = gl_SubgroupSize;
    mask = ZeroValueuint4();
    lsb = ZeroValueuint();
    msb = ZeroValueuint();
    uint _e8 = gl_SubgroupInvocationID_1;
    uint _e9 = gl_SubgroupSize_1;
    mask = uint4(_e8, _e9, 3u, 4u);
    uint4 _e14 = mask;
    const uint unnamed = _e14.x;
    const uint unnamed_1 = _e14.y;
    const uint unnamed_2 = _e14.z;
    const uint unnamed_3 = _e14.w;
    const uint _e15 = (unnamed != 0u ? firstbitlow(unnamed) : (unnamed_1 != 0u ? firstbitlow(unnamed_1) + 32u : (unnamed_2 != 0u ? firstbitlow(unnamed_2) + 64u : firstbitlow(unnamed_3) + 96u)));
    lsb = _e15;
    uint4 _e17 = mask;
    const uint unnamed_4 = _e17.x;
    const uint unnamed_5 = _e17.y;
    const uint unnamed_6 = _e17.z;
    const uint unnamed_7 = _e17.w;
    const uint _e18 = (unnamed_7 != 0u ? firstbithigh(unnamed_7) + 96u : (unnamed_6 != 0u ? firstbithigh(unnamed_6) + 64u : (unnamed_5 != 0u ? firstbithigh(unnamed_5) + 32u : firstbithigh(unnamed_4))));
    msb = _e18;
    uint _e20 = lsb;
    uint _e21 = msb;
    global.Store(0, asuint((_e20 + _e21)));
    return;
}
