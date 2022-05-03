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

Foo ConstructFoo(float4 arg0, int arg1) {
    Foo ret = (Foo)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

Foo Constructarray3_Foo_(Foo arg0, Foo arg1, Foo arg2)[3] {
    Foo ret[3] = { arg0, arg1, arg2 };
    return ret;
}

float4 builtins()
{
    int s1_ = (true ? 1 : 0);
    float4 s2_ = (true ? float4(1.0, 1.0, 1.0, 1.0) : float4(0.0, 0.0, 0.0, 0.0));
    float4 s3_ = (bool4(false, false, false, false) ? float4(0.0, 0.0, 0.0, 0.0) : float4(1.0, 1.0, 1.0, 1.0));
    float4 m1_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), float4(0.5, 0.5, 0.5, 0.5));
    float4 m2_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    float b1_ = float(int4(1, 1, 1, 1).x);
    float4 b2_ = float4(int4(1, 1, 1, 1));
    int4 v_i32_zero = int4(float4(0.0, 0.0, 0.0, 0.0));
    return (((((float4(((s1_).xxxx + v_i32_zero)) + s2_) + m1_) + m2_) + (b1_).xxxx) + b2_);
}

float4 splat()
{
    float2 a_1 = ((((1.0).xx + (2.0).xx) - (3.0).xx) / (4.0).xx);
    int4 b = ((5).xxxx % (2).xxxx);
    return (a_1.xyxy + float4(b));
}

float3 bool_cast(float3 x)
{
    bool3 y = bool3(x);
    return float3(y);
}

int Constructarray4_int_(int arg0, int arg1, int arg2, int arg3)[4] {
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
    float _expr59 = foo.a.x;
    return _expr59;
}

void logical()
{
    bool unnamed_3 = !true;
    bool2 unnamed_4 = !(true).xx;
    bool unnamed_5 = (true || false);
    bool unnamed_6 = (true && false);
    bool unnamed_7 = (true | false);
    bool3 unnamed_8 = ((true).xxx | (false).xxx);
    bool unnamed_9 = (true & false);
    bool4 unnamed_10 = ((true).xxxx & (false).xxxx);
}

void arithmetic()
{
    int2 unnamed_11 = -(1).xx;
    float2 unnamed_12 = -(1.0).xx;
    int unnamed_13 = (2 + 1);
    uint unnamed_14 = (2u + 1u);
    float unnamed_15 = (2.0 + 1.0);
    int2 unnamed_16 = ((2).xx + (1).xx);
    uint3 unnamed_17 = ((2u).xxx + (1u).xxx);
    float4 unnamed_18 = ((2.0).xxxx + (1.0).xxxx);
    int unnamed_19 = (2 - 1);
    uint unnamed_20 = (2u - 1u);
    float unnamed_21 = (2.0 - 1.0);
    int2 unnamed_22 = ((2).xx - (1).xx);
    uint3 unnamed_23 = ((2u).xxx - (1u).xxx);
    float4 unnamed_24 = ((2.0).xxxx - (1.0).xxxx);
    int unnamed_25 = (2 * 1);
    uint unnamed_26 = (2u * 1u);
    float unnamed_27 = (2.0 * 1.0);
    int2 unnamed_28 = ((2).xx * (1).xx);
    uint3 unnamed_29 = ((2u).xxx * (1u).xxx);
    float4 unnamed_30 = ((2.0).xxxx * (1.0).xxxx);
    int unnamed_31 = (2 / 1);
    uint unnamed_32 = (2u / 1u);
    float unnamed_33 = (2.0 / 1.0);
    int2 unnamed_34 = ((2).xx / (1).xx);
    uint3 unnamed_35 = ((2u).xxx / (1u).xxx);
    float4 unnamed_36 = ((2.0).xxxx / (1.0).xxxx);
    int unnamed_37 = (2 % 1);
    uint unnamed_38 = (2u % 1u);
    float unnamed_39 = (2.0 % 1.0);
    int2 unnamed_40 = ((2).xx % (1).xx);
    uint3 unnamed_41 = ((2u).xxx % (1u).xxx);
    float4 unnamed_42 = ((2.0).xxxx % (1.0).xxxx);
    int2 unnamed_43 = ((2).xx + (1).xx);
    int2 unnamed_44 = ((2).xx + (1).xx);
    uint2 unnamed_45 = ((2u).xx + (1u).xx);
    uint2 unnamed_46 = ((2u).xx + (1u).xx);
    float2 unnamed_47 = ((2.0).xx + (1.0).xx);
    float2 unnamed_48 = ((2.0).xx + (1.0).xx);
    int2 unnamed_49 = ((2).xx - (1).xx);
    int2 unnamed_50 = ((2).xx - (1).xx);
    uint2 unnamed_51 = ((2u).xx - (1u).xx);
    uint2 unnamed_52 = ((2u).xx - (1u).xx);
    float2 unnamed_53 = ((2.0).xx - (1.0).xx);
    float2 unnamed_54 = ((2.0).xx - (1.0).xx);
    int2 unnamed_55 = ((2).xx * 1);
    int2 unnamed_56 = (2 * (1).xx);
    uint2 unnamed_57 = ((2u).xx * 1u);
    uint2 unnamed_58 = (2u * (1u).xx);
    float2 unnamed_59 = ((2.0).xx * 1.0);
    float2 unnamed_60 = (2.0 * (1.0).xx);
    int2 unnamed_61 = ((2).xx / (1).xx);
    int2 unnamed_62 = ((2).xx / (1).xx);
    uint2 unnamed_63 = ((2u).xx / (1u).xx);
    uint2 unnamed_64 = ((2u).xx / (1u).xx);
    float2 unnamed_65 = ((2.0).xx / (1.0).xx);
    float2 unnamed_66 = ((2.0).xx / (1.0).xx);
    int2 unnamed_67 = ((2).xx % (1).xx);
    int2 unnamed_68 = ((2).xx % (1).xx);
    uint2 unnamed_69 = ((2u).xx % (1u).xx);
    uint2 unnamed_70 = ((2u).xx % (1u).xx);
    float2 unnamed_71 = ((2.0).xx % (1.0).xx);
    float2 unnamed_72 = ((2.0).xx % (1.0).xx);
    float3x3 unnamed_73 = (float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)) + float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
    float3x3 unnamed_74 = (float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)) - float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
    float3x3 unnamed_75 = mul(1.0, float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
    float3x3 unnamed_76 = mul(float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)), 2.0);
    float3 unnamed_77 = mul((1.0).xxxx, float4x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
    float4 unnamed_78 = mul(float4x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)), (2.0).xxx);
    float3x3 unnamed_79 = mul(float3x4(float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0)), float4x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
}

void bit()
{
    int unnamed_80 = ~1;
    uint unnamed_81 = ~1u;
    int2 unnamed_82 = ~(1).xx;
    uint3 unnamed_83 = ~(1u).xxx;
    int unnamed_84 = (2 | 1);
    uint unnamed_85 = (2u | 1u);
    int2 unnamed_86 = ((2).xx | (1).xx);
    uint3 unnamed_87 = ((2u).xxx | (1u).xxx);
    int unnamed_88 = (2 & 1);
    uint unnamed_89 = (2u & 1u);
    int2 unnamed_90 = ((2).xx & (1).xx);
    uint3 unnamed_91 = ((2u).xxx & (1u).xxx);
    int unnamed_92 = (2 ^ 1);
    uint unnamed_93 = (2u ^ 1u);
    int2 unnamed_94 = ((2).xx ^ (1).xx);
    uint3 unnamed_95 = ((2u).xxx ^ (1u).xxx);
    int unnamed_96 = (2 << 1u);
    uint unnamed_97 = (2u << 1u);
    int2 unnamed_98 = ((2).xx << (1u).xx);
    uint3 unnamed_99 = ((2u).xxx << (1u).xxx);
    int unnamed_100 = (2 >> 1u);
    uint unnamed_101 = (2u >> 1u);
    int2 unnamed_102 = ((2).xx >> (1u).xx);
    uint3 unnamed_103 = ((2u).xxx >> (1u).xxx);
}

void comparison()
{
    bool unnamed_104 = (2 == 1);
    bool unnamed_105 = (2u == 1u);
    bool unnamed_106 = (2.0 == 1.0);
    bool2 unnamed_107 = ((2).xx == (1).xx);
    bool3 unnamed_108 = ((2u).xxx == (1u).xxx);
    bool4 unnamed_109 = ((2.0).xxxx == (1.0).xxxx);
    bool unnamed_110 = (2 != 1);
    bool unnamed_111 = (2u != 1u);
    bool unnamed_112 = (2.0 != 1.0);
    bool2 unnamed_113 = ((2).xx != (1).xx);
    bool3 unnamed_114 = ((2u).xxx != (1u).xxx);
    bool4 unnamed_115 = ((2.0).xxxx != (1.0).xxxx);
    bool unnamed_116 = (2 < 1);
    bool unnamed_117 = (2u < 1u);
    bool unnamed_118 = (2.0 < 1.0);
    bool2 unnamed_119 = ((2).xx < (1).xx);
    bool3 unnamed_120 = ((2u).xxx < (1u).xxx);
    bool4 unnamed_121 = ((2.0).xxxx < (1.0).xxxx);
    bool unnamed_122 = (2 <= 1);
    bool unnamed_123 = (2u <= 1u);
    bool unnamed_124 = (2.0 <= 1.0);
    bool2 unnamed_125 = ((2).xx <= (1).xx);
    bool3 unnamed_126 = ((2u).xxx <= (1u).xxx);
    bool4 unnamed_127 = ((2.0).xxxx <= (1.0).xxxx);
    bool unnamed_128 = (2 > 1);
    bool unnamed_129 = (2u > 1u);
    bool unnamed_130 = (2.0 > 1.0);
    bool2 unnamed_131 = ((2).xx > (1).xx);
    bool3 unnamed_132 = ((2u).xxx > (1u).xxx);
    bool4 unnamed_133 = ((2.0).xxxx > (1.0).xxxx);
    bool unnamed_134 = (2 >= 1);
    bool unnamed_135 = (2u >= 1u);
    bool unnamed_136 = (2.0 >= 1.0);
    bool2 unnamed_137 = ((2).xx >= (1).xx);
    bool3 unnamed_138 = ((2u).xxx >= (1u).xxx);
    bool4 unnamed_139 = ((2.0).xxxx >= (1.0).xxxx);
}

void assignment()
{
    int a = 1;

    int _expr6 = a;
    a = (_expr6 + 1);
    int _expr9 = a;
    a = (_expr9 - 1);
    int _expr12 = a;
    int _expr13 = a;
    a = (_expr12 * _expr13);
    int _expr15 = a;
    int _expr16 = a;
    a = (_expr15 / _expr16);
    int _expr18 = a;
    a = (_expr18 % 1);
    int _expr21 = a;
    a = (_expr21 & 0);
    int _expr24 = a;
    a = (_expr24 | 0);
    int _expr27 = a;
    a = (_expr27 ^ 0);
    int _expr30 = a;
    a = (_expr30 << 2u);
    int _expr33 = a;
    a = (_expr33 >> 1u);
    int _expr36 = a;
    a = (_expr36 + 1);
    int _expr39 = a;
    a = (_expr39 - 1);
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e4 = builtins();
    const float4 _e5 = splat();
    const float3 _e7 = bool_cast(float4(1.0, 1.0, 1.0, 1.0).xyz);
    const float _e8 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
