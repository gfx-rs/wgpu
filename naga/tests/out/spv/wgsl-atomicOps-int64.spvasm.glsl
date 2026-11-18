#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    uint64_t _m0;
    int64_t _m1[2];
};

struct _10
{
    uint64_t _m0;
    int64_t _m1[2];
};

struct _13
{
    uint64_t _m0;
    bool _m1;
};

struct _14
{
    int64_t _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _16_15
{
    uint64_t _m0;
} _15;

layout(set = 0, binding = 1, std430) buffer _19_18
{
    int64_t _m0[2];
} _18;

layout(set = 0, binding = 2, std430) buffer _22_21
{
    _9 _m0;
} _21;

shared uint64_t _24;
shared int64_t _26[2];
shared _10 _28;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _24 = 0ul;
        _26 = int64_t[](0l, 0l);
        _28 = _10(0ul, int64_t[](0l, 0l));
    }
    barrier();
    atomicExchange(_15._m0, 1ul);
    atomicExchange(_18._m0[1u], 1l);
    atomicExchange(_21._m0._m0, 1ul);
    atomicExchange(_21._m0._m1[1u], 1l);
    atomicExchange(_24, 1ul);
    atomicExchange(_26[1u], 1l);
    atomicExchange(_28._m0, 1ul);
    atomicExchange(_28._m1[1u], 1l);
    barrier();
    uint64_t _71 = atomicAdd(_15._m0, 0);
    int64_t _73 = atomicAdd(_18._m0[1u], 0);
    uint64_t _75 = atomicAdd(_21._m0._m0, 0);
    int64_t _77 = atomicAdd(_21._m0._m1[1u], 0);
    uint64_t _78 = atomicAdd(_24, 0);
    int64_t _80 = atomicAdd(_26[1u], 0);
    uint64_t _82 = atomicAdd(_28._m0, 0);
    int64_t _84 = atomicAdd(_28._m1[1u], 0);
    barrier();
    uint64_t _85 = atomicAdd(_15._m0, 1ul);
    int64_t _86 = atomicAdd(_18._m0[1u], 1l);
    uint64_t _88 = atomicAdd(_21._m0._m0, 1ul);
    int64_t _90 = atomicAdd(_21._m0._m1[1u], 1l);
    uint64_t _92 = atomicAdd(_24, 1ul);
    int64_t _93 = atomicAdd(_26[1u], 1l);
    uint64_t _95 = atomicAdd(_28._m0, 1ul);
    int64_t _97 = atomicAdd(_28._m1[1u], 1l);
    barrier();
    uint64_t _99 = atomicAdd(_15._m0, -1ul);
    int64_t _100 = atomicAdd(_18._m0[1u], -1l);
    uint64_t _102 = atomicAdd(_21._m0._m0, -1ul);
    int64_t _104 = atomicAdd(_21._m0._m1[1u], -1l);
    uint64_t _106 = atomicAdd(_24, -1ul);
    int64_t _107 = atomicAdd(_26[1u], -1l);
    uint64_t _109 = atomicAdd(_28._m0, -1ul);
    int64_t _111 = atomicAdd(_28._m1[1u], -1l);
    barrier();
    uint64_t _113 = atomicMax(_15._m0, 1ul);
    int64_t _114 = atomicMax(_18._m0[1u], 1l);
    uint64_t _116 = atomicMax(_21._m0._m0, 1ul);
    int64_t _118 = atomicMax(_21._m0._m1[1u], 1l);
    uint64_t _120 = atomicMax(_24, 1ul);
    int64_t _121 = atomicMax(_26[1u], 1l);
    uint64_t _123 = atomicMax(_28._m0, 1ul);
    int64_t _125 = atomicMax(_28._m1[1u], 1l);
    barrier();
    uint64_t _127 = atomicMin(_15._m0, 1ul);
    int64_t _128 = atomicMin(_18._m0[1u], 1l);
    uint64_t _130 = atomicMin(_21._m0._m0, 1ul);
    int64_t _132 = atomicMin(_21._m0._m1[1u], 1l);
    uint64_t _134 = atomicMin(_24, 1ul);
    int64_t _135 = atomicMin(_26[1u], 1l);
    uint64_t _137 = atomicMin(_28._m0, 1ul);
    int64_t _139 = atomicMin(_28._m1[1u], 1l);
    barrier();
    uint64_t _141 = atomicAnd(_15._m0, 1ul);
    int64_t _142 = atomicAnd(_18._m0[1u], 1l);
    uint64_t _144 = atomicAnd(_21._m0._m0, 1ul);
    int64_t _146 = atomicAnd(_21._m0._m1[1u], 1l);
    uint64_t _148 = atomicAnd(_24, 1ul);
    int64_t _149 = atomicAnd(_26[1u], 1l);
    uint64_t _151 = atomicAnd(_28._m0, 1ul);
    int64_t _153 = atomicAnd(_28._m1[1u], 1l);
    barrier();
    uint64_t _155 = atomicOr(_15._m0, 1ul);
    int64_t _156 = atomicOr(_18._m0[1u], 1l);
    uint64_t _158 = atomicOr(_21._m0._m0, 1ul);
    int64_t _160 = atomicOr(_21._m0._m1[1u], 1l);
    uint64_t _162 = atomicOr(_24, 1ul);
    int64_t _163 = atomicOr(_26[1u], 1l);
    uint64_t _165 = atomicOr(_28._m0, 1ul);
    int64_t _167 = atomicOr(_28._m1[1u], 1l);
    barrier();
    uint64_t _169 = atomicXor(_15._m0, 1ul);
    int64_t _170 = atomicXor(_18._m0[1u], 1l);
    uint64_t _172 = atomicXor(_21._m0._m0, 1ul);
    int64_t _174 = atomicXor(_21._m0._m1[1u], 1l);
    uint64_t _176 = atomicXor(_24, 1ul);
    int64_t _177 = atomicXor(_26[1u], 1l);
    uint64_t _179 = atomicXor(_28._m0, 1ul);
    int64_t _181 = atomicXor(_28._m1[1u], 1l);
    uint64_t _183 = atomicExchange(_15._m0, 1ul);
    int64_t _184 = atomicExchange(_18._m0[1u], 1l);
    uint64_t _186 = atomicExchange(_21._m0._m0, 1ul);
    int64_t _188 = atomicExchange(_21._m0._m1[1u], 1l);
    uint64_t _190 = atomicExchange(_24, 1ul);
    int64_t _191 = atomicExchange(_26[1u], 1l);
    uint64_t _193 = atomicExchange(_28._m0, 1ul);
    int64_t _195 = atomicExchange(_28._m1[1u], 1l);
    uint64_t _198 = atomicCompSwap(_15._m0, 1ul, 2ul);
    int64_t _202 = atomicCompSwap(_18._m0[1u], 1l, 2l);
    uint64_t _206 = atomicCompSwap(_21._m0._m0, 1ul, 2ul);
    int64_t _210 = atomicCompSwap(_21._m0._m1[1u], 1l, 2l);
    uint64_t _213 = atomicCompSwap(_24, 1ul, 2ul);
    int64_t _217 = atomicCompSwap(_26[1u], 1l, 2l);
    uint64_t _221 = atomicCompSwap(_28._m0, 1ul, 2ul);
    int64_t _225 = atomicCompSwap(_28._m1[1u], 1l, 2l);
}

