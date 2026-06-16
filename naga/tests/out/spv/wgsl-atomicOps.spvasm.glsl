#version 460
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    uint _m0;
    int _m1[2];
};

struct _10
{
    uint _m0;
    bool _m1;
};

struct _11
{
    int _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _13_12
{
    uint _m0;
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    int _m0[2];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    _7 _m0;
} _18;

shared uint _21;
shared int _23[2];
shared _7 _25;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _21 = 0u;
        _23 = int[](0, 0);
        _25 = _7(0u, int[](0, 0));
    }
    barrier();
    atomicExchange(_12._m0, 1u);
    atomicExchange(_15._m0[1u], 1);
    atomicExchange(_18._m0._m0, 1u);
    atomicExchange(_18._m0._m1[1u], 1);
    atomicExchange(_21, 1u);
    atomicExchange(_23[1u], 1);
    atomicExchange(_25._m0, 1u);
    atomicExchange(_25._m1[1u], 1);
    barrier();
    uint _63 = atomicAdd(_12._m0, 0u);
    int _65 = atomicAdd(_15._m0[1u], 0);
    uint _67 = atomicAdd(_18._m0._m0, 0u);
    int _69 = atomicAdd(_18._m0._m1[1u], 0);
    uint _70 = atomicAdd(_21, 0u);
    int _72 = atomicAdd(_23[1u], 0);
    uint _74 = atomicAdd(_25._m0, 0u);
    int _76 = atomicAdd(_25._m1[1u], 0);
    barrier();
    uint _77 = atomicAdd(_12._m0, 1u);
    int _78 = atomicAdd(_15._m0[1u], 1);
    uint _80 = atomicAdd(_18._m0._m0, 1u);
    int _82 = atomicAdd(_18._m0._m1[1u], 1);
    uint _84 = atomicAdd(_21, 1u);
    int _85 = atomicAdd(_23[1u], 1);
    uint _87 = atomicAdd(_25._m0, 1u);
    int _89 = atomicAdd(_25._m1[1u], 1);
    barrier();
    uint _91 = atomicAdd(_12._m0, -1u);
    int _92 = atomicAdd(_15._m0[1u], -1);
    uint _94 = atomicAdd(_18._m0._m0, -1u);
    int _96 = atomicAdd(_18._m0._m1[1u], -1);
    uint _98 = atomicAdd(_21, -1u);
    int _99 = atomicAdd(_23[1u], -1);
    uint _101 = atomicAdd(_25._m0, -1u);
    int _103 = atomicAdd(_25._m1[1u], -1);
    barrier();
    uint _105 = atomicMax(_12._m0, 1u);
    int _106 = atomicMax(_15._m0[1u], 1);
    uint _108 = atomicMax(_18._m0._m0, 1u);
    int _110 = atomicMax(_18._m0._m1[1u], 1);
    uint _112 = atomicMax(_21, 1u);
    int _113 = atomicMax(_23[1u], 1);
    uint _115 = atomicMax(_25._m0, 1u);
    int _117 = atomicMax(_25._m1[1u], 1);
    barrier();
    uint _119 = atomicMin(_12._m0, 1u);
    int _120 = atomicMin(_15._m0[1u], 1);
    uint _122 = atomicMin(_18._m0._m0, 1u);
    int _124 = atomicMin(_18._m0._m1[1u], 1);
    uint _126 = atomicMin(_21, 1u);
    int _127 = atomicMin(_23[1u], 1);
    uint _129 = atomicMin(_25._m0, 1u);
    int _131 = atomicMin(_25._m1[1u], 1);
    barrier();
    uint _133 = atomicAnd(_12._m0, 1u);
    int _134 = atomicAnd(_15._m0[1u], 1);
    uint _136 = atomicAnd(_18._m0._m0, 1u);
    int _138 = atomicAnd(_18._m0._m1[1u], 1);
    uint _140 = atomicAnd(_21, 1u);
    int _141 = atomicAnd(_23[1u], 1);
    uint _143 = atomicAnd(_25._m0, 1u);
    int _145 = atomicAnd(_25._m1[1u], 1);
    barrier();
    uint _147 = atomicOr(_12._m0, 1u);
    int _148 = atomicOr(_15._m0[1u], 1);
    uint _150 = atomicOr(_18._m0._m0, 1u);
    int _152 = atomicOr(_18._m0._m1[1u], 1);
    uint _154 = atomicOr(_21, 1u);
    int _155 = atomicOr(_23[1u], 1);
    uint _157 = atomicOr(_25._m0, 1u);
    int _159 = atomicOr(_25._m1[1u], 1);
    barrier();
    uint _161 = atomicXor(_12._m0, 1u);
    int _162 = atomicXor(_15._m0[1u], 1);
    uint _164 = atomicXor(_18._m0._m0, 1u);
    int _166 = atomicXor(_18._m0._m1[1u], 1);
    uint _168 = atomicXor(_21, 1u);
    int _169 = atomicXor(_23[1u], 1);
    uint _171 = atomicXor(_25._m0, 1u);
    int _173 = atomicXor(_25._m1[1u], 1);
    uint _175 = atomicExchange(_12._m0, 1u);
    int _176 = atomicExchange(_15._m0[1u], 1);
    uint _178 = atomicExchange(_18._m0._m0, 1u);
    int _180 = atomicExchange(_18._m0._m1[1u], 1);
    uint _182 = atomicExchange(_21, 1u);
    int _183 = atomicExchange(_23[1u], 1);
    uint _185 = atomicExchange(_25._m0, 1u);
    int _187 = atomicExchange(_25._m1[1u], 1);
    uint _190 = atomicCompSwap(_12._m0, 1u, 2u);
    int _194 = atomicCompSwap(_15._m0[1u], 1, 2);
    uint _198 = atomicCompSwap(_18._m0._m0, 1u, 2u);
    int _202 = atomicCompSwap(_18._m0._m1[1u], 1, 2);
    uint _205 = atomicCompSwap(_21, 1u, 2u);
    int _209 = atomicCompSwap(_23[1u], 1, 2);
    uint _213 = atomicCompSwap(_25._m0, 1u, 2u);
    int _217 = atomicCompSwap(_25._m1[1u], 1, 2);
}

