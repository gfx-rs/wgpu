#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    uint64_t _m0;
    uint64_t _m1[2];
};

layout(set = 0, binding = 0, std430) buffer _10_9
{
    uint64_t _m0;
} _9;

layout(set = 0, binding = 1, std430) buffer _13_12
{
    uint64_t _m0[2];
} _12;

layout(set = 0, binding = 2, std430) buffer _16_15
{
    _7 _m0;
} _15;

layout(set = 0, binding = 3, std140) uniform _19_18
{
    uint64_t _m0;
} _18;

void main()
{
    uint64_t _39 = atomicMax(_9._m0, _18._m0);
    uint64_t _44 = atomicMax(_12._m0[1u], 1ul + _18._m0);
    uint64_t _47 = atomicMax(_15._m0._m0, 1ul);
    uint64_t _51 = atomicMax(_15._m0._m1[1u], uint64_t(gl_LocalInvocationID.x));
    barrier();
    uint64_t _55 = atomicMin(_9._m0, _18._m0);
    uint64_t _58 = atomicMin(_12._m0[1u], 1ul + _18._m0);
    uint64_t _60 = atomicMin(_15._m0._m0, 1ul);
    uint64_t _64 = atomicMin(_15._m0._m1[1u], uint64_t(gl_LocalInvocationID.x));
}

