struct Output {
    uint result;
};

RWByteAddressBuffer global : register(u0);
static uint3 gl_GlobalInvocationID_1 = (uint3)0;

void main_1()
{
    uint4 mask = (uint4)0;
    uint lsb = (uint)0;
    uint msb = (uint)0;

    uint3 _e3 = gl_GlobalInvocationID_1;
    mask = uint4(_e3.x, 2u, 3u, 4u);
    uint4 _e10 = mask;
    const uint _e11 = (_e10.x != 0u ? firstbitlow(_e10.x) : (_e10.y != 0u ? firstbitlow(_e10.y) + 32u : (_e10.z != 0u ? firstbitlow(_e10.z) + 64u : firstbitlow(_e10.w) + 96u)));
    lsb = _e11;
    uint4 _e13 = mask;
    const uint _e14 = (_e13.w != 0u ? firstbithigh(_e13.w) + 96u : (_e13.z != 0u ? firstbithigh(_e13.z) + 64u : (_e13.y != 0u ? firstbithigh(_e13.y) + 32u : firstbithigh(_e13.x))));
    msb = _e14;
    uint _e16 = lsb;
    uint _e17 = msb;
    global.Store(0, asuint((_e16 + _e17)));
    return;
}

[numthreads(1, 1, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID)
{
    gl_GlobalInvocationID_1 = gl_GlobalInvocationID;
    main_1();
    return;
}
