////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_i64" (comp) //
////////////////////////////////////////////////////////////
#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    int64_t _m0;
    bool _m1;
};

struct _11
{
    uint64_t _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _14_13
{
    int64_t _m0[128];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    uint64_t _m0[128];
} _16;

void main()
{
    uvec2 _42 = uvec2(0u);
    int64_t _33 = 0l;
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
        int64_t _69 = atomicAdd(_13._m0[_31], 0);
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
            int64_t _95 = _33;
            int64_t _98 = atomicCompSwap(_13._m0[_31], _95, _33 + 10l);
            _10 _96 = _10(_98, _98 == _95);
            _33 = _96._m0;
            _36 = _96._m1;
            continue;
        }
        _31++;
        continue;
    }
}


////////////////////////////////////////////////////////////
// Entry point: "test_atomic_compare_exchange_u64" (comp) //
////////////////////////////////////////////////////////////
#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    int64_t _m0;
    bool _m1;
};

struct _11
{
    uint64_t _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _14_13
{
    int64_t _m0[128];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    uint64_t _m0[128];
} _16;

void main()
{
    uvec2 _117 = uvec2(0u);
    uint64_t _110 = 0ul;
    uvec2 _115 = uvec2(0u);
    uint _109 = 0u;
    bool _113 = false;
    _115 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_115, uvec2(0u))))
        {
            break;
        }
        _115 -= uvec2(uint(_115.y == 0u), 1u);
        if (!(_109 < 128u))
        {
            break;
        }
        uint64_t _143 = atomicAdd(_16._m0[_109], 0);
        _110 = _143;
        _113 = false;
        _117 = uvec2(4294967295u);
        for (;;)
        {
            if (all(equal(_117, uvec2(0u))))
            {
                break;
            }
            _117 -= uvec2(uint(_117.y == 0u), 1u);
            if (!(!_113))
            {
                break;
            }
            uint64_t _167 = _110;
            uint64_t _170 = atomicCompSwap(_16._m0[_109], _167, _110 + 10ul);
            _11 _168 = _11(_170, _170 == _167);
            _110 = _168._m0;
            _113 = _168._m1;
            continue;
        }
        _109++;
        continue;
    }
}

