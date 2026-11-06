struct Output {
    uint result;
};

RWByteAddressBuffer global : register(u0);
static uint gl_SubgroupInvocationID_1 = (uint)0;
static uint gl_SubgroupSize_1 = (uint)0;

struct ComputeInput_main {
};

void main_1()
{
    uint4 mask = (uint4)0;
    uint lsb = (uint)0;
    uint msb = (uint)0;

    uint _e4 = gl_SubgroupInvocationID_1;
    uint _e5 = gl_SubgroupSize_1;
    mask = uint4(_e4, _e5, 3u, 4u);
    uint4 _e10 = mask;
    const uint unnamed = _e10.x;
    const uint unnamed_1 = _e10.y;
    const uint unnamed_2 = _e10.z;
    const uint unnamed_3 = _e10.w;
    const uint _e11 = (unnamed != 0u ? firstbitlow(unnamed) : (unnamed_1 != 0u ? firstbitlow(unnamed_1) + 32u : (unnamed_2 != 0u ? firstbitlow(unnamed_2) + 64u : firstbitlow(unnamed_3) + 96u)));
    lsb = _e11;
    uint4 _e13 = mask;
    const uint unnamed_4 = _e13.x;
    const uint unnamed_5 = _e13.y;
    const uint unnamed_6 = _e13.z;
    const uint unnamed_7 = _e13.w;
    const uint _e14 = (unnamed_7 != 0u ? firstbithigh(unnamed_7) + 96u : (unnamed_6 != 0u ? firstbithigh(unnamed_6) + 64u : (unnamed_5 != 0u ? firstbithigh(unnamed_5) + 32u : firstbithigh(unnamed_4))));
    msb = _e14;
    uint _e16 = lsb;
    uint _e17 = msb;
    global.Store(0, asuint((_e16 + _e17)));
    return;
}

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint gl_SubgroupInvocationID = WaveGetLaneIndex();
    uint gl_SubgroupSize = WaveGetLaneCount();
    gl_SubgroupInvocationID_1 = gl_SubgroupInvocationID;
    gl_SubgroupSize_1 = gl_SubgroupSize;
    main_1();
    return;
}
