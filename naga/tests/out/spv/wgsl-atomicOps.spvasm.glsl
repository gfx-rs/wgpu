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
    for (uint _54 = gl_LocalInvocationIndex; _54 < 2u; _54 += 2u)
    {
        _23[_54] = 0;
    }
    for (uint _63 = gl_LocalInvocationIndex; _63 < 2u; _63 += 2u)
    {
        _25._m1[_63] = 0;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _21 = 0u;
        _25._m0 = 0u;
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
    uint _79 = atomicAdd(_12._m0, 0u);
    int _81 = atomicAdd(_15._m0[1u], 0);
    uint _83 = atomicAdd(_18._m0._m0, 0u);
    int _85 = atomicAdd(_18._m0._m1[1u], 0);
    uint _86 = atomicAdd(_21, 0u);
    int _88 = atomicAdd(_23[1u], 0);
    uint _90 = atomicAdd(_25._m0, 0u);
    int _92 = atomicAdd(_25._m1[1u], 0);
    barrier();
    uint _93 = atomicAdd(_12._m0, 1u);
    int _94 = atomicAdd(_15._m0[1u], 1);
    uint _96 = atomicAdd(_18._m0._m0, 1u);
    int _98 = atomicAdd(_18._m0._m1[1u], 1);
    uint _100 = atomicAdd(_21, 1u);
    int _101 = atomicAdd(_23[1u], 1);
    uint _103 = atomicAdd(_25._m0, 1u);
    int _105 = atomicAdd(_25._m1[1u], 1);
    barrier();
    uint _107 = atomicAdd(_12._m0, -1u);
    int _108 = atomicAdd(_15._m0[1u], -1);
    uint _110 = atomicAdd(_18._m0._m0, -1u);
    int _112 = atomicAdd(_18._m0._m1[1u], -1);
    uint _114 = atomicAdd(_21, -1u);
    int _115 = atomicAdd(_23[1u], -1);
    uint _117 = atomicAdd(_25._m0, -1u);
    int _119 = atomicAdd(_25._m1[1u], -1);
    barrier();
    uint _121 = atomicMax(_12._m0, 1u);
    int _122 = atomicMax(_15._m0[1u], 1);
    uint _124 = atomicMax(_18._m0._m0, 1u);
    int _126 = atomicMax(_18._m0._m1[1u], 1);
    uint _128 = atomicMax(_21, 1u);
    int _129 = atomicMax(_23[1u], 1);
    uint _131 = atomicMax(_25._m0, 1u);
    int _133 = atomicMax(_25._m1[1u], 1);
    barrier();
    uint _135 = atomicMin(_12._m0, 1u);
    int _136 = atomicMin(_15._m0[1u], 1);
    uint _138 = atomicMin(_18._m0._m0, 1u);
    int _140 = atomicMin(_18._m0._m1[1u], 1);
    uint _142 = atomicMin(_21, 1u);
    int _143 = atomicMin(_23[1u], 1);
    uint _145 = atomicMin(_25._m0, 1u);
    int _147 = atomicMin(_25._m1[1u], 1);
    barrier();
    uint _149 = atomicAnd(_12._m0, 1u);
    int _150 = atomicAnd(_15._m0[1u], 1);
    uint _152 = atomicAnd(_18._m0._m0, 1u);
    int _154 = atomicAnd(_18._m0._m1[1u], 1);
    uint _156 = atomicAnd(_21, 1u);
    int _157 = atomicAnd(_23[1u], 1);
    uint _159 = atomicAnd(_25._m0, 1u);
    int _161 = atomicAnd(_25._m1[1u], 1);
    barrier();
    uint _163 = atomicOr(_12._m0, 1u);
    int _164 = atomicOr(_15._m0[1u], 1);
    uint _166 = atomicOr(_18._m0._m0, 1u);
    int _168 = atomicOr(_18._m0._m1[1u], 1);
    uint _170 = atomicOr(_21, 1u);
    int _171 = atomicOr(_23[1u], 1);
    uint _173 = atomicOr(_25._m0, 1u);
    int _175 = atomicOr(_25._m1[1u], 1);
    barrier();
    uint _177 = atomicXor(_12._m0, 1u);
    int _178 = atomicXor(_15._m0[1u], 1);
    uint _180 = atomicXor(_18._m0._m0, 1u);
    int _182 = atomicXor(_18._m0._m1[1u], 1);
    uint _184 = atomicXor(_21, 1u);
    int _185 = atomicXor(_23[1u], 1);
    uint _187 = atomicXor(_25._m0, 1u);
    int _189 = atomicXor(_25._m1[1u], 1);
    uint _191 = atomicExchange(_12._m0, 1u);
    int _192 = atomicExchange(_15._m0[1u], 1);
    uint _194 = atomicExchange(_18._m0._m0, 1u);
    int _196 = atomicExchange(_18._m0._m1[1u], 1);
    uint _198 = atomicExchange(_21, 1u);
    int _199 = atomicExchange(_23[1u], 1);
    uint _201 = atomicExchange(_25._m0, 1u);
    int _203 = atomicExchange(_25._m1[1u], 1);
    uint _206 = atomicCompSwap(_12._m0, 1u, 2u);
    int _210 = atomicCompSwap(_15._m0[1u], 1, 2);
    uint _214 = atomicCompSwap(_18._m0._m0, 1u, 2u);
    int _218 = atomicCompSwap(_18._m0._m1[1u], 1, 2);
    uint _221 = atomicCompSwap(_21, 1u, 2u);
    int _225 = atomicCompSwap(_23[1u], 1, 2);
    uint _229 = atomicCompSwap(_25._m0, 1u, 2u);
    int _233 = atomicCompSwap(_25._m1[1u], 1, 2);
}

