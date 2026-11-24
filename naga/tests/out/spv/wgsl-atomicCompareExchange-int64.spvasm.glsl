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
    uvec2 _41 = uvec2(4294967295u);
    int64_t _33 = 0l;
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
        int64_t _69 = atomicAdd(_13._m0[_31], 0);
        _33 = _69;
        _36 = false;
        _41 = uvec2(4294967295u);
        for (;;)
        {
            uvec2 _78 = _41;
            _41 = _78 - uvec2(uint(_78.y == 0u), 1u);
            if (all(equal(_78, uvec2(0u))))
            {
                break;
            }
            if (!(!_36))
            {
                break;
            }
            int64_t _97 = _33;
            int64_t _100 = atomicCompSwap(_13._m0[_31], _97, _33 + 10l);
            _10 _98 = _10(_100, _100 == _97);
            _33 = _98._m0;
            _36 = _98._m1;
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
    uvec2 _118 = uvec2(4294967295u);
    uint64_t _112 = 0ul;
    uvec2 _117 = uvec2(4294967295u);
    uint _111 = 0u;
    bool _115 = false;
    for (;;)
    {
        uvec2 _126 = _117;
        _117 = _126 - uvec2(uint(_126.y == 0u), 1u);
        if (all(equal(_126, uvec2(0u))))
        {
            break;
        }
        if (!(_111 < 128u))
        {
            break;
        }
        uint64_t _145 = atomicAdd(_16._m0[_111], 0);
        _112 = _145;
        _115 = false;
        _118 = uvec2(4294967295u);
        for (;;)
        {
            uvec2 _152 = _118;
            _118 = _152 - uvec2(uint(_152.y == 0u), 1u);
            if (all(equal(_152, uvec2(0u))))
            {
                break;
            }
            if (!(!_115))
            {
                break;
            }
            uint64_t _171 = _112;
            uint64_t _174 = atomicCompSwap(_16._m0[_111], _171, _112 + 10ul);
            _11 _172 = _11(_174, _174 == _171);
            _112 = _172._m0;
            _115 = _172._m1;
            continue;
        }
        _111++;
        continue;
    }
}

