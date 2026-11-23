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
    uvec2 _42 = uvec2(0u);
    int _33 = 0;
    uvec2 _39 = uvec2(0u);
    uint _31 = 0u;
    bool _36 = false;
    _39 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_39, uvec2(0u))))
        {
            break;
        }
        _39 -= uvec2(uint(_39.y == 0u), 1u);
        if (!(_31 < 128u))
        {
            break;
        }
        int _69 = atomicAdd(_13._m0[_31], 0);
        _33 = _69;
        _36 = false;
        _42 = uvec2(4294967295u);
        for (;;)
        {
            if (all(equal(_42, uvec2(0u))))
            {
                break;
            }
            _42 -= uvec2(uint(_42.y == 0u), 1u);
            if (!(!_36))
            {
                break;
            }
            int _96 = _33;
            int _99 = atomicCompSwap(_13._m0[_31], _96, floatBitsToInt(intBitsToFloat(_33) + 1.0));
            _10 _97 = _10(_99, _99 == _96);
            _33 = _97._m0;
            _36 = _97._m1;
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
    uvec2 _116 = uvec2(0u);
    uint _110 = 0u;
    uvec2 _114 = uvec2(0u);
    uint _109 = 0u;
    bool _112 = false;
    _114 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_114, uvec2(0u))))
        {
            break;
        }
        _114 -= uvec2(uint(_114.y == 0u), 1u);
        if (!(_109 < 128u))
        {
            break;
        }
        uint _142 = atomicAdd(_16._m0[_109], 0u);
        _110 = _142;
        _112 = false;
        _116 = uvec2(4294967295u);
        for (;;)
        {
            if (all(equal(_116, uvec2(0u))))
            {
                break;
            }
            _116 -= uvec2(uint(_116.y == 0u), 1u);
            if (!(!_112))
            {
                break;
            }
            uint _168 = _110;
            uint _171 = atomicCompSwap(_16._m0[_109], _168, floatBitsToUint(uintBitsToFloat(_110) + 1.0));
            _11 _169 = _11(_171, _171 == _168);
            _110 = _169._m0;
            _112 = _169._m1;
            continue;
        }
        _109++;
        continue;
    }
}

