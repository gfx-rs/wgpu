////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_i32" (comp) //
////////////////////////////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    int _m0;
    bool _m1;
};

struct _10
{
    uint _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _12_11
{
    int _m0[128];
} _11;

layout(set = 0, binding = 1, std430) buffer _15_14
{
    uint _m0[128];
} _14;

void main()
{
    uint _27 = 0u;
    int _29 = 0;
    bool _32 = false;
    uvec2 _46 = uvec2(4294967295u);
    uvec2 _72 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _46)))
        {
            break;
        }
        _46 -= uvec2(uint(_46.y == 0u), 1u);
        if (!(_27 < 128u))
        {
            break;
        }
        int _66 = atomicAdd(_11._m0[_27], 0);
        _29 = _66;
        _32 = false;
        for (;;)
        {
            if (all(equal(uvec2(0u), _72)))
            {
                break;
            }
            _72 -= uvec2(uint(_72.y == 0u), 1u);
            if (!(!_32))
            {
                break;
            }
            int _94 = _29;
            int _97 = atomicCompSwap(_11._m0[_27], _94, floatBitsToInt(intBitsToFloat(_29) + 1.0));
            _9 _95 = _9(_97, _97 == _94);
            _29 = _95._m0;
            _32 = _95._m1;
            continue;
        }
        _27++;
        continue;
    }
}


////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_u32" (comp) //
////////////////////////////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    int _m0;
    bool _m1;
};

struct _10
{
    uint _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _12_11
{
    int _m0[128];
} _11;

layout(set = 0, binding = 1, std430) buffer _15_14
{
    uint _m0[128];
} _14;

void main()
{
    uint _107 = 0u;
    uint _108 = 0u;
    bool _110 = false;
    uvec2 _117 = uvec2(4294967295u);
    uvec2 _142 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _117)))
        {
            break;
        }
        _117 -= uvec2(uint(_117.y == 0u), 1u);
        if (!(_107 < 128u))
        {
            break;
        }
        uint _137 = atomicAdd(_14._m0[_107], 0u);
        _108 = _137;
        _110 = false;
        for (;;)
        {
            if (all(equal(uvec2(0u), _142)))
            {
                break;
            }
            _142 -= uvec2(uint(_142.y == 0u), 1u);
            if (!(!_110))
            {
                break;
            }
            uint _164 = _108;
            uint _167 = atomicCompSwap(_14._m0[_107], _164, floatBitsToUint(uintBitsToFloat(_108) + 1.0));
            _10 _165 = _10(_167, _167 == _164);
            _108 = _165._m0;
            _110 = _165._m1;
            continue;
        }
        _107++;
        continue;
    }
}

