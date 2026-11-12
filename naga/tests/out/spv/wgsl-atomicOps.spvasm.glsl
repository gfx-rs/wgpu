#version 460
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _8
{
    uint _m0;
    int _m1[2];
};

struct _9
{
    uint _m0;
    int _m1[2];
};

struct _12
{
    uint _m0;
    bool _m1;
};

struct _13
{
    int _m0;
    bool _m1;
};

layout(set = 0, binding = 0, std430) buffer _15_14
{
    uint _m0;
} _14;

layout(set = 0, binding = 1, std430) buffer _18_17
{
    int _m0[2];
} _17;

layout(set = 0, binding = 2, std430) buffer _21_20
{
    _8 _m0;
} _20;

shared uint _23;
shared int _25[2];
shared _9 _27;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _23 = 0u;
        _25 = int[](0, 0);
        _27 = _9(0u, int[](0, 0));
    }
    barrier();
    atomicExchange(_14._m0, 1u);
    atomicExchange(_17._m0[1u], 1);
    atomicExchange(_20._m0._m0, 1u);
    atomicExchange(_20._m0._m1[1u], 1);
    atomicExchange(_23, 1u);
    atomicExchange(_25[1u], 1);
    atomicExchange(_27._m0, 1u);
    atomicExchange(_27._m1[1u], 1);
    barrier();
    uint _65 = atomicAdd(_14._m0, 0u);
    int _67 = atomicAdd(_17._m0[1u], 0);
    uint _69 = atomicAdd(_20._m0._m0, 0u);
    int _71 = atomicAdd(_20._m0._m1[1u], 0);
    uint _72 = atomicAdd(_23, 0u);
    int _74 = atomicAdd(_25[1u], 0);
    uint _76 = atomicAdd(_27._m0, 0u);
    int _78 = atomicAdd(_27._m1[1u], 0);
    barrier();
    uint _79 = atomicAdd(_14._m0, 1u);
    int _80 = atomicAdd(_17._m0[1u], 1);
    uint _82 = atomicAdd(_20._m0._m0, 1u);
    int _84 = atomicAdd(_20._m0._m1[1u], 1);
    uint _86 = atomicAdd(_23, 1u);
    int _87 = atomicAdd(_25[1u], 1);
    uint _89 = atomicAdd(_27._m0, 1u);
    int _91 = atomicAdd(_27._m1[1u], 1);
    barrier();
    uint _93 = atomicAdd(_14._m0, -1u);
    int _94 = atomicAdd(_17._m0[1u], -1);
    uint _96 = atomicAdd(_20._m0._m0, -1u);
    int _98 = atomicAdd(_20._m0._m1[1u], -1);
    uint _100 = atomicAdd(_23, -1u);
    int _101 = atomicAdd(_25[1u], -1);
    uint _103 = atomicAdd(_27._m0, -1u);
    int _105 = atomicAdd(_27._m1[1u], -1);
    barrier();
    uint _107 = atomicMax(_14._m0, 1u);
    int _108 = atomicMax(_17._m0[1u], 1);
    uint _110 = atomicMax(_20._m0._m0, 1u);
    int _112 = atomicMax(_20._m0._m1[1u], 1);
    uint _114 = atomicMax(_23, 1u);
    int _115 = atomicMax(_25[1u], 1);
    uint _117 = atomicMax(_27._m0, 1u);
    int _119 = atomicMax(_27._m1[1u], 1);
    barrier();
    uint _121 = atomicMin(_14._m0, 1u);
    int _122 = atomicMin(_17._m0[1u], 1);
    uint _124 = atomicMin(_20._m0._m0, 1u);
    int _126 = atomicMin(_20._m0._m1[1u], 1);
    uint _128 = atomicMin(_23, 1u);
    int _129 = atomicMin(_25[1u], 1);
    uint _131 = atomicMin(_27._m0, 1u);
    int _133 = atomicMin(_27._m1[1u], 1);
    barrier();
    uint _135 = atomicAnd(_14._m0, 1u);
    int _136 = atomicAnd(_17._m0[1u], 1);
    uint _138 = atomicAnd(_20._m0._m0, 1u);
    int _140 = atomicAnd(_20._m0._m1[1u], 1);
    uint _142 = atomicAnd(_23, 1u);
    int _143 = atomicAnd(_25[1u], 1);
    uint _145 = atomicAnd(_27._m0, 1u);
    int _147 = atomicAnd(_27._m1[1u], 1);
    barrier();
    uint _149 = atomicOr(_14._m0, 1u);
    int _150 = atomicOr(_17._m0[1u], 1);
    uint _152 = atomicOr(_20._m0._m0, 1u);
    int _154 = atomicOr(_20._m0._m1[1u], 1);
    uint _156 = atomicOr(_23, 1u);
    int _157 = atomicOr(_25[1u], 1);
    uint _159 = atomicOr(_27._m0, 1u);
    int _161 = atomicOr(_27._m1[1u], 1);
    barrier();
    uint _163 = atomicXor(_14._m0, 1u);
    int _164 = atomicXor(_17._m0[1u], 1);
    uint _166 = atomicXor(_20._m0._m0, 1u);
    int _168 = atomicXor(_20._m0._m1[1u], 1);
    uint _170 = atomicXor(_23, 1u);
    int _171 = atomicXor(_25[1u], 1);
    uint _173 = atomicXor(_27._m0, 1u);
    int _175 = atomicXor(_27._m1[1u], 1);
    uint _177 = atomicExchange(_14._m0, 1u);
    int _178 = atomicExchange(_17._m0[1u], 1);
    uint _180 = atomicExchange(_20._m0._m0, 1u);
    int _182 = atomicExchange(_20._m0._m1[1u], 1);
    uint _184 = atomicExchange(_23, 1u);
    int _185 = atomicExchange(_25[1u], 1);
    uint _187 = atomicExchange(_27._m0, 1u);
    int _189 = atomicExchange(_27._m1[1u], 1);
    uint _192 = atomicCompSwap(_14._m0, 1u, 2u);
    int _196 = atomicCompSwap(_17._m0[1u], 1, 2);
    uint _200 = atomicCompSwap(_20._m0._m0, 1u, 2u);
    int _204 = atomicCompSwap(_20._m0._m1[1u], 1, 2);
    uint _207 = atomicCompSwap(_23, 1u, 2u);
    int _211 = atomicCompSwap(_25[1u], 1, 2);
    uint _215 = atomicCompSwap(_27._m0, 1u, 2u);
    int _219 = atomicCompSwap(_27._m1[1u], 1, 2);
}

