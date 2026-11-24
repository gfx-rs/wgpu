////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_i32" (comp) //
////////////////////////////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    int _m0;
    bool _m1;
};

struct _11
{
    uint _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _14_13
{
    int _m0[128];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    uint _m0[128];
} _16;

void main()
{
    uvec2 _41 = uvec2(4294967295u);
    int _33 = 0;
    uvec2 _39 = uvec2(4294967295u);
    uint _31 = 0u;
    bool _36 = false;
    for (;;)
    {
        uvec2 _49 = _39;
        _39 = _49 - uvec2(uint(_49.y == 0u), 1u);
        if (all(equal(_49, uvec2(0u))))
        {
            break;
        }
        if (!(_31 < 128u))
        {
            break;
        }
        int _69 = atomicAdd(_13._m0[_31], 0);
        _33 = _69;
        _36 = false;
        _41 = uvec2(4294967295u);
        for (;;)
        {
            uvec2 _77 = _41;
            _41 = _77 - uvec2(uint(_77.y == 0u), 1u);
            if (all(equal(_77, uvec2(0u))))
            {
                break;
            }
            if (!(!_36))
            {
                break;
            }
            int _98 = _33;
            int _101 = atomicCompSwap(_13._m0[_31], _98, floatBitsToInt(intBitsToFloat(_33) + 1.0));
            _10 _99 = _10(_101, _101 == _98);
            _33 = _99._m0;
            _36 = _99._m1;
            continue;
        }
        _31++;
        continue;
    }
}


////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_u32" (comp) //
////////////////////////////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    int _m0;
    bool _m1;
};

struct _11
{
    uint _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _14_13
{
    int _m0[128];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    uint _m0[128];
} _16;

void main()
{
    uvec2 _117 = uvec2(4294967295u);
    uint _112 = 0u;
    uvec2 _116 = uvec2(4294967295u);
    uint _111 = 0u;
    bool _114 = false;
    for (;;)
    {
        uvec2 _125 = _116;
        _116 = _125 - uvec2(uint(_125.y == 0u), 1u);
        if (all(equal(_125, uvec2(0u))))
        {
            break;
        }
        if (!(_111 < 128u))
        {
            break;
        }
        uint _144 = atomicAdd(_16._m0[_111], 0u);
        _112 = _144;
        _114 = false;
        _117 = uvec2(4294967295u);
        for (;;)
        {
            uvec2 _151 = _117;
            _117 = _151 - uvec2(uint(_151.y == 0u), 1u);
            if (all(equal(_151, uvec2(0u))))
            {
                break;
            }
            if (!(!_114))
            {
                break;
            }
            uint _172 = _112;
            uint _175 = atomicCompSwap(_16._m0[_111], _172, floatBitsToUint(uintBitsToFloat(_112) + 1.0));
            _11 _173 = _11(_175, _175 == _172);
            _112 = _173._m0;
            _114 = _173._m1;
            continue;
        }
        _111++;
        continue;
    }
}

