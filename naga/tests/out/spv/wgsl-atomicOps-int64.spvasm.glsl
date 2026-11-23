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
    _24[gl_LocalInvocationIndex] = 0l;
    _26._m1[gl_LocalInvocationIndex] = 0l;
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
    uint64_t _70 = atomicAdd(_13._m0, 0);
    int64_t _72 = atomicAdd(_16._m0[1u], 0);
    uint64_t _74 = atomicAdd(_19._m0._m0, 0);
    int64_t _76 = atomicAdd(_19._m0._m1[1u], 0);
    uint64_t _77 = atomicAdd(_22, 0);
    int64_t _79 = atomicAdd(_24[1u], 0);
    uint64_t _81 = atomicAdd(_26._m0, 0);
    int64_t _83 = atomicAdd(_26._m1[1u], 0);
    barrier();
    uint64_t _84 = atomicAdd(_13._m0, 1ul);
    int64_t _85 = atomicAdd(_16._m0[1u], 1l);
    uint64_t _87 = atomicAdd(_19._m0._m0, 1ul);
    int64_t _89 = atomicAdd(_19._m0._m1[1u], 1l);
    uint64_t _91 = atomicAdd(_22, 1ul);
    int64_t _92 = atomicAdd(_24[1u], 1l);
    uint64_t _94 = atomicAdd(_26._m0, 1ul);
    int64_t _96 = atomicAdd(_26._m1[1u], 1l);
    barrier();
    uint64_t _98 = atomicAdd(_13._m0, -1ul);
    int64_t _99 = atomicAdd(_16._m0[1u], -1l);
    uint64_t _101 = atomicAdd(_19._m0._m0, -1ul);
    int64_t _103 = atomicAdd(_19._m0._m1[1u], -1l);
    uint64_t _105 = atomicAdd(_22, -1ul);
    int64_t _106 = atomicAdd(_24[1u], -1l);
    uint64_t _108 = atomicAdd(_26._m0, -1ul);
    int64_t _110 = atomicAdd(_26._m1[1u], -1l);
    barrier();
    uint64_t _112 = atomicMax(_13._m0, 1ul);
    int64_t _113 = atomicMax(_16._m0[1u], 1l);
    uint64_t _115 = atomicMax(_19._m0._m0, 1ul);
    int64_t _117 = atomicMax(_19._m0._m1[1u], 1l);
    uint64_t _119 = atomicMax(_22, 1ul);
    int64_t _120 = atomicMax(_24[1u], 1l);
    uint64_t _122 = atomicMax(_26._m0, 1ul);
    int64_t _124 = atomicMax(_26._m1[1u], 1l);
    barrier();
    uint64_t _126 = atomicMin(_13._m0, 1ul);
    int64_t _127 = atomicMin(_16._m0[1u], 1l);
    uint64_t _129 = atomicMin(_19._m0._m0, 1ul);
    int64_t _131 = atomicMin(_19._m0._m1[1u], 1l);
    uint64_t _133 = atomicMin(_22, 1ul);
    int64_t _134 = atomicMin(_24[1u], 1l);
    uint64_t _136 = atomicMin(_26._m0, 1ul);
    int64_t _138 = atomicMin(_26._m1[1u], 1l);
    barrier();
    uint64_t _140 = atomicAnd(_13._m0, 1ul);
    int64_t _141 = atomicAnd(_16._m0[1u], 1l);
    uint64_t _143 = atomicAnd(_19._m0._m0, 1ul);
    int64_t _145 = atomicAnd(_19._m0._m1[1u], 1l);
    uint64_t _147 = atomicAnd(_22, 1ul);
    int64_t _148 = atomicAnd(_24[1u], 1l);
    uint64_t _150 = atomicAnd(_26._m0, 1ul);
    int64_t _152 = atomicAnd(_26._m1[1u], 1l);
    barrier();
    uint64_t _154 = atomicOr(_13._m0, 1ul);
    int64_t _155 = atomicOr(_16._m0[1u], 1l);
    uint64_t _157 = atomicOr(_19._m0._m0, 1ul);
    int64_t _159 = atomicOr(_19._m0._m1[1u], 1l);
    uint64_t _161 = atomicOr(_22, 1ul);
    int64_t _162 = atomicOr(_24[1u], 1l);
    uint64_t _164 = atomicOr(_26._m0, 1ul);
    int64_t _166 = atomicOr(_26._m1[1u], 1l);
    barrier();
    uint64_t _168 = atomicXor(_13._m0, 1ul);
    int64_t _169 = atomicXor(_16._m0[1u], 1l);
    uint64_t _171 = atomicXor(_19._m0._m0, 1ul);
    int64_t _173 = atomicXor(_19._m0._m1[1u], 1l);
    uint64_t _175 = atomicXor(_22, 1ul);
    int64_t _176 = atomicXor(_24[1u], 1l);
    uint64_t _178 = atomicXor(_26._m0, 1ul);
    int64_t _180 = atomicXor(_26._m1[1u], 1l);
    uint64_t _182 = atomicExchange(_13._m0, 1ul);
    int64_t _183 = atomicExchange(_16._m0[1u], 1l);
    uint64_t _185 = atomicExchange(_19._m0._m0, 1ul);
    int64_t _187 = atomicExchange(_19._m0._m1[1u], 1l);
    uint64_t _189 = atomicExchange(_22, 1ul);
    int64_t _190 = atomicExchange(_24[1u], 1l);
    uint64_t _192 = atomicExchange(_26._m0, 1ul);
    int64_t _194 = atomicExchange(_26._m1[1u], 1l);
    uint64_t _197 = atomicCompSwap(_13._m0, 1ul, 2ul);
    int64_t _201 = atomicCompSwap(_16._m0[1u], 1l, 2l);
    uint64_t _205 = atomicCompSwap(_19._m0._m0, 1ul, 2ul);
    int64_t _209 = atomicCompSwap(_19._m0._m1[1u], 1l, 2l);
    uint64_t _212 = atomicCompSwap(_22, 1ul, 2ul);
    int64_t _216 = atomicCompSwap(_24[1u], 1l, 2l);
    uint64_t _220 = atomicCompSwap(_26._m0, 1ul, 2ul);
    int64_t _224 = atomicCompSwap(_26._m1[1u], 1l, 2l);
}

