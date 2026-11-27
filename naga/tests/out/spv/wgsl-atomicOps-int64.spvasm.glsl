#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _8
{
    uint64_t _m0;
    int64_t _m1[2];
};

struct _11
{
    uint64_t _m0;
    bool _m1;
};

struct _12
{
    int64_t _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _14_13
{
    uint64_t _m0;
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    int64_t _m0[2];
} _16;

layout(set = 0, binding = 2, std430) buffer _20_19
{
    _8 _m0;
} _19;

shared uint64_t _22;
shared int64_t _24[2];
shared _8 _26;

void main()
{
    for (uint _56 = gl_LocalInvocationIndex; _56 < 2u; _56 += 2u)
    {
        _24[_56] = 0l;
    }
    for (uint _65 = gl_LocalInvocationIndex; _65 < 2u; _65 += 2u)
    {
        _26._m1[_65] = 0l;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _22 = 0ul;
        _26._m0 = 0ul;
    }
    barrier();
    atomicExchange(_13._m0, 1ul);
    atomicExchange(_16._m0[1u], 1l);
    atomicExchange(_19._m0._m0, 1ul);
    atomicExchange(_19._m0._m1[1u], 1l);
    atomicExchange(_22, 1ul);
    atomicExchange(_24[1u], 1l);
    atomicExchange(_26._m0, 1ul);
    atomicExchange(_26._m1[1u], 1l);
    barrier();
    uint64_t _85 = atomicAdd(_13._m0, 0);
    int64_t _87 = atomicAdd(_16._m0[1u], 0);
    uint64_t _89 = atomicAdd(_19._m0._m0, 0);
    int64_t _91 = atomicAdd(_19._m0._m1[1u], 0);
    uint64_t _92 = atomicAdd(_22, 0);
    int64_t _94 = atomicAdd(_24[1u], 0);
    uint64_t _96 = atomicAdd(_26._m0, 0);
    int64_t _98 = atomicAdd(_26._m1[1u], 0);
    barrier();
    uint64_t _99 = atomicAdd(_13._m0, 1ul);
    int64_t _100 = atomicAdd(_16._m0[1u], 1l);
    uint64_t _102 = atomicAdd(_19._m0._m0, 1ul);
    int64_t _104 = atomicAdd(_19._m0._m1[1u], 1l);
    uint64_t _106 = atomicAdd(_22, 1ul);
    int64_t _107 = atomicAdd(_24[1u], 1l);
    uint64_t _109 = atomicAdd(_26._m0, 1ul);
    int64_t _111 = atomicAdd(_26._m1[1u], 1l);
    barrier();
    uint64_t _113 = atomicAdd(_13._m0, -1ul);
    int64_t _114 = atomicAdd(_16._m0[1u], -1l);
    uint64_t _116 = atomicAdd(_19._m0._m0, -1ul);
    int64_t _118 = atomicAdd(_19._m0._m1[1u], -1l);
    uint64_t _120 = atomicAdd(_22, -1ul);
    int64_t _121 = atomicAdd(_24[1u], -1l);
    uint64_t _123 = atomicAdd(_26._m0, -1ul);
    int64_t _125 = atomicAdd(_26._m1[1u], -1l);
    barrier();
    uint64_t _127 = atomicMax(_13._m0, 1ul);
    int64_t _128 = atomicMax(_16._m0[1u], 1l);
    uint64_t _130 = atomicMax(_19._m0._m0, 1ul);
    int64_t _132 = atomicMax(_19._m0._m1[1u], 1l);
    uint64_t _134 = atomicMax(_22, 1ul);
    int64_t _135 = atomicMax(_24[1u], 1l);
    uint64_t _137 = atomicMax(_26._m0, 1ul);
    int64_t _139 = atomicMax(_26._m1[1u], 1l);
    barrier();
    uint64_t _141 = atomicMin(_13._m0, 1ul);
    int64_t _142 = atomicMin(_16._m0[1u], 1l);
    uint64_t _144 = atomicMin(_19._m0._m0, 1ul);
    int64_t _146 = atomicMin(_19._m0._m1[1u], 1l);
    uint64_t _148 = atomicMin(_22, 1ul);
    int64_t _149 = atomicMin(_24[1u], 1l);
    uint64_t _151 = atomicMin(_26._m0, 1ul);
    int64_t _153 = atomicMin(_26._m1[1u], 1l);
    barrier();
    uint64_t _155 = atomicAnd(_13._m0, 1ul);
    int64_t _156 = atomicAnd(_16._m0[1u], 1l);
    uint64_t _158 = atomicAnd(_19._m0._m0, 1ul);
    int64_t _160 = atomicAnd(_19._m0._m1[1u], 1l);
    uint64_t _162 = atomicAnd(_22, 1ul);
    int64_t _163 = atomicAnd(_24[1u], 1l);
    uint64_t _165 = atomicAnd(_26._m0, 1ul);
    int64_t _167 = atomicAnd(_26._m1[1u], 1l);
    barrier();
    uint64_t _169 = atomicOr(_13._m0, 1ul);
    int64_t _170 = atomicOr(_16._m0[1u], 1l);
    uint64_t _172 = atomicOr(_19._m0._m0, 1ul);
    int64_t _174 = atomicOr(_19._m0._m1[1u], 1l);
    uint64_t _176 = atomicOr(_22, 1ul);
    int64_t _177 = atomicOr(_24[1u], 1l);
    uint64_t _179 = atomicOr(_26._m0, 1ul);
    int64_t _181 = atomicOr(_26._m1[1u], 1l);
    barrier();
    uint64_t _183 = atomicXor(_13._m0, 1ul);
    int64_t _184 = atomicXor(_16._m0[1u], 1l);
    uint64_t _186 = atomicXor(_19._m0._m0, 1ul);
    int64_t _188 = atomicXor(_19._m0._m1[1u], 1l);
    uint64_t _190 = atomicXor(_22, 1ul);
    int64_t _191 = atomicXor(_24[1u], 1l);
    uint64_t _193 = atomicXor(_26._m0, 1ul);
    int64_t _195 = atomicXor(_26._m1[1u], 1l);
    uint64_t _197 = atomicExchange(_13._m0, 1ul);
    int64_t _198 = atomicExchange(_16._m0[1u], 1l);
    uint64_t _200 = atomicExchange(_19._m0._m0, 1ul);
    int64_t _202 = atomicExchange(_19._m0._m1[1u], 1l);
    uint64_t _204 = atomicExchange(_22, 1ul);
    int64_t _205 = atomicExchange(_24[1u], 1l);
    uint64_t _207 = atomicExchange(_26._m0, 1ul);
    int64_t _209 = atomicExchange(_26._m1[1u], 1l);
    uint64_t _212 = atomicCompSwap(_13._m0, 1ul, 2ul);
    int64_t _216 = atomicCompSwap(_16._m0[1u], 1l, 2l);
    uint64_t _220 = atomicCompSwap(_19._m0._m0, 1ul, 2ul);
    int64_t _224 = atomicCompSwap(_19._m0._m1[1u], 1l, 2l);
    uint64_t _227 = atomicCompSwap(_22, 1ul, 2ul);
    int64_t _231 = atomicCompSwap(_24[1u], 1l, 2l);
    uint64_t _235 = atomicCompSwap(_26._m0, 1ul, 2ul);
    int64_t _239 = atomicCompSwap(_26._m1[1u], 1l, 2l);
}

