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

layout(set = 0, binding = 0, std430) buffer _13_12
{
    int64_t _m0[128];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    uint64_t _m0[128];
} _15;

void main()
{
    uint _27 = 0u;
    int64_t _29 = 0l;
    bool _32 = false;
    uvec2 _46 = uvec2(4294967295u);
    uvec2 _73 = uvec2(4294967295u);
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
        int64_t _66 = atomicAdd(_12._m0[_27], 0);
        _29 = _66;
        _32 = false;
        for (;;)
        {
            if (all(equal(uvec2(0u), _73)))
            {
                break;
            }
            _73 -= uvec2(uint(_73.y == 0u), 1u);
            if (!(!_32))
            {
                break;
            }
            int64_t _93 = _29;
            int64_t _96 = atomicCompSwap(_12._m0[_27], _93, _29 + 10l);
            _10 _94 = _10(_96, _96 == _93);
            _29 = _94._m0;
            _32 = _94._m1;
            continue;
        }
        _27++;
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

layout(set = 0, binding = 0, std430) buffer _13_12
{
    int64_t _m0[128];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    uint64_t _m0[128];
} _15;

void main()
{
    uint _107 = 0u;
    uint64_t _108 = 0ul;
    bool _111 = false;
    uvec2 _118 = uvec2(4294967295u);
    uvec2 _143 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _118)))
        {
            break;
        }
        _118 -= uvec2(uint(_118.y == 0u), 1u);
        if (!(_107 < 128u))
        {
            break;
        }
        uint64_t _138 = atomicAdd(_15._m0[_107], 0);
        _108 = _138;
        _111 = false;
        for (;;)
        {
            if (all(equal(uvec2(0u), _143)))
            {
                break;
            }
            _143 -= uvec2(uint(_143.y == 0u), 1u);
            if (!(!_111))
            {
                break;
            }
            uint64_t _163 = _108;
            uint64_t _166 = atomicCompSwap(_15._m0[_107], _163, _108 + 10ul);
            _11 _164 = _11(_166, _166 == _163);
            _108 = _164._m0;
            _111 = _164._m1;
            continue;
        }
        _107++;
        continue;
    }
}

