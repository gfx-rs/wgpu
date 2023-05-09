static const float4 v_f32_one = float4(1.0, 1.0, 1.0, 1.0);
static const float4 v_f32_zero = float4(0.0, 0.0, 0.0, 0.0);
static const float4 v_f32_half = float4(0.5, 0.5, 0.5, 0.5);
static const int4 v_i32_one = int4(1, 1, 1, 1);

struct Foo {
    float4 a;
    int b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

float4 builtins()
{
    int s1_ = (true ? 1 : 0);
    float4 s2_ = (true ? float4(1.0, 1.0, 1.0, 1.0) : float4(0.0, 0.0, 0.0, 0.0));
    float4 s3_ = (bool4(false, false, false, false) ? float4(0.0, 0.0, 0.0, 0.0) : float4(1.0, 1.0, 1.0, 1.0));
    float4 m1_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), float4(0.5, 0.5, 0.5, 0.5));
    float4 m2_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    float b1_ = asfloat(int4(1, 1, 1, 1).x);
    float4 b2_ = asfloat(int4(1, 1, 1, 1));
    int4 v_i32_zero = int4(float4(0.0, 0.0, 0.0, 0.0));
    return (((((float4(((s1_).xxxx + v_i32_zero)) + s2_) + m1_) + m2_) + (b1_).xxxx) + b2_);
}

float4 splat()
{
    float2 a_2 = ((((1.0).xx + (2.0).xx) - (3.0).xx) / (4.0).xx);
    int4 b = ((5).xxxx % (2).xxxx);
    return (a_2.xyxy + float4(b));
}

float2 splat_assignment()
{
    float2 a = (float2)0;

    a = (2.0).xx;
    float2 _expr4 = a;
    a = (_expr4 + (1.0).xx);
    float2 _expr8 = a;
    a = (_expr8 - (3.0).xx);
    float2 _expr12 = a;
    a = (_expr12 / (4.0).xx);
    float2 _expr15 = a;
    return _expr15;
}

float3 bool_cast(float3 x)
{
    bool3 y = bool3(x);
    return float3(y);
}

Foo ConstructFoo(float4 arg0, int arg1) {
    Foo ret = (Foo)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

typedef int ret_Constructarray4_int_[4];
ret_Constructarray4_int_ Constructarray4_int_(int arg0, int arg1, int arg2, int arg3) {
    int ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

float constructors()
{
    Foo foo = (Foo)0;

    foo = ConstructFoo((1.0).xxxx, 1);
    float2x2 mat2comp = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    float4x4 mat4comp = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    uint2 unnamed = (0u).xx;
    float2x2 unnamed_1 = float2x2((0.0).xx, (0.0).xx);
    int unnamed_2[4] = Constructarray4_int_(0, 1, 2, 3);
    bool unnamed_3 = bool((bool)0);
    int unnamed_4 = int((int)0);
    uint unnamed_5 = uint((uint)0);
    float unnamed_6 = float((float)0);
    uint2 unnamed_7 = uint2((uint2)0);
    float2x3 unnamed_8 = float2x3((float2x3)0);
    uint2 unnamed_9 = asuint((uint2)0);
    float2x3 unnamed_10 = asfloat((float2x3)0);
    float _expr71 = foo.a.x;
    return _expr71;
}

void logical()
{
    bool unnamed_11 = !(true);
    bool2 unnamed_12 = !((true).xx);
    bool unnamed_13 = (true || false);
    bool unnamed_14 = (true && false);
    bool unnamed_15 = (true | false);
    bool3 unnamed_16 = ((true).xxx | (false).xxx);
    bool unnamed_17 = (true & false);
    bool4 unnamed_18 = ((true).xxxx & (false).xxxx);
}

void arithmetic()
{
    int2 unnamed_19 = -((1).xx);
    float2 unnamed_20 = -((1.0).xx);
    int unnamed_21 = (2 + 1);
    uint unnamed_22 = (2u + 1u);
    float unnamed_23 = (2.0 + 1.0);
    int2 unnamed_24 = ((2).xx + (1).xx);
    uint3 unnamed_25 = ((2u).xxx + (1u).xxx);
    float4 unnamed_26 = ((2.0).xxxx + (1.0).xxxx);
    int unnamed_27 = (2 - 1);
    uint unnamed_28 = (2u - 1u);
    float unnamed_29 = (2.0 - 1.0);
    int2 unnamed_30 = ((2).xx - (1).xx);
    uint3 unnamed_31 = ((2u).xxx - (1u).xxx);
    float4 unnamed_32 = ((2.0).xxxx - (1.0).xxxx);
    int unnamed_33 = (2 * 1);
    uint unnamed_34 = (2u * 1u);
    float unnamed_35 = (2.0 * 1.0);
    int2 unnamed_36 = ((2).xx * (1).xx);
    uint3 unnamed_37 = ((2u).xxx * (1u).xxx);
    float4 unnamed_38 = ((2.0).xxxx * (1.0).xxxx);
    int unnamed_39 = (2 / 1);
    uint unnamed_40 = (2u / 1u);
    float unnamed_41 = (2.0 / 1.0);
    int2 unnamed_42 = ((2).xx / (1).xx);
    uint3 unnamed_43 = ((2u).xxx / (1u).xxx);
    float4 unnamed_44 = ((2.0).xxxx / (1.0).xxxx);
    int unnamed_45 = (2 % 1);
    uint unnamed_46 = (2u % 1u);
    float unnamed_47 = fmod(2.0, 1.0);
    int2 unnamed_48 = ((2).xx % (1).xx);
    uint3 unnamed_49 = ((2u).xxx % (1u).xxx);
    float4 unnamed_50 = fmod((2.0).xxxx, (1.0).xxxx);
    int2 unnamed_51 = ((2).xx + (1).xx);
    int2 unnamed_52 = ((2).xx + (1).xx);
    uint2 unnamed_53 = ((2u).xx + (1u).xx);
    uint2 unnamed_54 = ((2u).xx + (1u).xx);
    float2 unnamed_55 = ((2.0).xx + (1.0).xx);
    float2 unnamed_56 = ((2.0).xx + (1.0).xx);
    int2 unnamed_57 = ((2).xx - (1).xx);
    int2 unnamed_58 = ((2).xx - (1).xx);
    uint2 unnamed_59 = ((2u).xx - (1u).xx);
    uint2 unnamed_60 = ((2u).xx - (1u).xx);
    float2 unnamed_61 = ((2.0).xx - (1.0).xx);
    float2 unnamed_62 = ((2.0).xx - (1.0).xx);
    int2 unnamed_63 = ((2).xx * 1);
    int2 unnamed_64 = (2 * (1).xx);
    uint2 unnamed_65 = ((2u).xx * 1u);
    uint2 unnamed_66 = (2u * (1u).xx);
    float2 unnamed_67 = ((2.0).xx * 1.0);
    float2 unnamed_68 = (2.0 * (1.0).xx);
    int2 unnamed_69 = ((2).xx / (1).xx);
    int2 unnamed_70 = ((2).xx / (1).xx);
    uint2 unnamed_71 = ((2u).xx / (1u).xx);
    uint2 unnamed_72 = ((2u).xx / (1u).xx);
    float2 unnamed_73 = ((2.0).xx / (1.0).xx);
    float2 unnamed_74 = ((2.0).xx / (1.0).xx);
    int2 unnamed_75 = ((2).xx % (1).xx);
    int2 unnamed_76 = ((2).xx % (1).xx);
    uint2 unnamed_77 = ((2u).xx % (1u).xx);
    uint2 unnamed_78 = ((2u).xx % (1u).xx);
    float2 unnamed_79 = fmod((2.0).xx, (1.0).xx);
    float2 unnamed_80 = fmod((2.0).xx, (1.0).xx);
    float3x3 unnamed_81 = ((float3x3)0 + (float3x3)0);
    float3x3 unnamed_82 = ((float3x3)0 - (float3x3)0);
    float3x3 unnamed_83 = mul(1.0, (float3x3)0);
    float3x3 unnamed_84 = mul((float3x3)0, 2.0);
    float3 unnamed_85 = mul((1.0).xxxx, (float4x3)0);
    float4 unnamed_86 = mul((float4x3)0, (2.0).xxx);
    float3x3 unnamed_87 = mul((float3x4)0, (float4x3)0);
}

void bit()
{
    int unnamed_88 = ~(1);
    uint unnamed_89 = ~(1u);
    int2 unnamed_90 = ~((1).xx);
    uint3 unnamed_91 = ~((1u).xxx);
    int unnamed_92 = (2 | 1);
    uint unnamed_93 = (2u | 1u);
    int2 unnamed_94 = ((2).xx | (1).xx);
    uint3 unnamed_95 = ((2u).xxx | (1u).xxx);
    int unnamed_96 = (2 & 1);
    uint unnamed_97 = (2u & 1u);
    int2 unnamed_98 = ((2).xx & (1).xx);
    uint3 unnamed_99 = ((2u).xxx & (1u).xxx);
    int unnamed_100 = (2 ^ 1);
    uint unnamed_101 = (2u ^ 1u);
    int2 unnamed_102 = ((2).xx ^ (1).xx);
    uint3 unnamed_103 = ((2u).xxx ^ (1u).xxx);
    int unnamed_104 = (2 << 1u);
    uint unnamed_105 = (2u << 1u);
    int2 unnamed_106 = ((2).xx << (1u).xx);
    uint3 unnamed_107 = ((2u).xxx << (1u).xxx);
    int unnamed_108 = (2 >> 1u);
    uint unnamed_109 = (2u >> 1u);
    int2 unnamed_110 = ((2).xx >> (1u).xx);
    uint3 unnamed_111 = ((2u).xxx >> (1u).xxx);
}

void comparison()
{
    bool unnamed_112 = (2 == 1);
    bool unnamed_113 = (2u == 1u);
    bool unnamed_114 = (2.0 == 1.0);
    bool2 unnamed_115 = ((2).xx == (1).xx);
    bool3 unnamed_116 = ((2u).xxx == (1u).xxx);
    bool4 unnamed_117 = ((2.0).xxxx == (1.0).xxxx);
    bool unnamed_118 = (2 != 1);
    bool unnamed_119 = (2u != 1u);
    bool unnamed_120 = (2.0 != 1.0);
    bool2 unnamed_121 = ((2).xx != (1).xx);
    bool3 unnamed_122 = ((2u).xxx != (1u).xxx);
    bool4 unnamed_123 = ((2.0).xxxx != (1.0).xxxx);
    bool unnamed_124 = (2 < 1);
    bool unnamed_125 = (2u < 1u);
    bool unnamed_126 = (2.0 < 1.0);
    bool2 unnamed_127 = ((2).xx < (1).xx);
    bool3 unnamed_128 = ((2u).xxx < (1u).xxx);
    bool4 unnamed_129 = ((2.0).xxxx < (1.0).xxxx);
    bool unnamed_130 = (2 <= 1);
    bool unnamed_131 = (2u <= 1u);
    bool unnamed_132 = (2.0 <= 1.0);
    bool2 unnamed_133 = ((2).xx <= (1).xx);
    bool3 unnamed_134 = ((2u).xxx <= (1u).xxx);
    bool4 unnamed_135 = ((2.0).xxxx <= (1.0).xxxx);
    bool unnamed_136 = (2 > 1);
    bool unnamed_137 = (2u > 1u);
    bool unnamed_138 = (2.0 > 1.0);
    bool2 unnamed_139 = ((2).xx > (1).xx);
    bool3 unnamed_140 = ((2u).xxx > (1u).xxx);
    bool4 unnamed_141 = ((2.0).xxxx > (1.0).xxxx);
    bool unnamed_142 = (2 >= 1);
    bool unnamed_143 = (2u >= 1u);
    bool unnamed_144 = (2.0 >= 1.0);
    bool2 unnamed_145 = ((2).xx >= (1).xx);
    bool3 unnamed_146 = ((2u).xxx >= (1u).xxx);
    bool4 unnamed_147 = ((2.0).xxxx >= (1.0).xxxx);
}

void assignment()
{
    int a_1 = (int)0;
    int3 vec0_ = (int3)0;

    a_1 = 1;
    int _expr3 = a_1;
    a_1 = (_expr3 + 1);
    int _expr6 = a_1;
    a_1 = (_expr6 - 1);
    int _expr8 = a_1;
    int _expr9 = a_1;
    a_1 = (_expr9 * _expr8);
    int _expr11 = a_1;
    int _expr12 = a_1;
    a_1 = (_expr12 / _expr11);
    int _expr15 = a_1;
    a_1 = (_expr15 % 1);
    int _expr18 = a_1;
    a_1 = (_expr18 & 0);
    int _expr21 = a_1;
    a_1 = (_expr21 | 0);
    int _expr24 = a_1;
    a_1 = (_expr24 ^ 0);
    int _expr27 = a_1;
    a_1 = (_expr27 << 2u);
    int _expr30 = a_1;
    a_1 = (_expr30 >> 1u);
    int _expr32 = a_1;
    a_1 = (_expr32 + 1);
    int _expr35 = a_1;
    a_1 = (_expr35 - 1);
    vec0_ = (int3)0;
    int _expr42 = vec0_.y;
    vec0_.y = (_expr42 + 1);
    int _expr47 = vec0_.y;
    vec0_.y = (_expr47 - 1);
    return;
}

void negation_avoids_prefix_decrement()
{
    int unnamed_148 = -(-2);
    int unnamed_149 = -(-3);
    int unnamed_150 = -(-(4));
    int unnamed_151 = -(-(-5));
    int unnamed_152 = -(-(-(-(6))));
    int unnamed_153 = -(-(-(-(-7))));
    int unnamed_154 = -(-(-(-(-8))));
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e0 = builtins();
    const float4 _e1 = splat();
    const float3 _e4 = bool_cast(float4(1.0, 1.0, 1.0, 1.0).xyz);
    const float _e5 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
