struct Foo {
    float4 a;
    int b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

static const float4 v_f32_one = float4(1.0, 1.0, 1.0, 1.0);
static const float4 v_f32_zero = float4(0.0, 0.0, 0.0, 0.0);
static const float4 v_f32_half = float4(0.5, 0.5, 0.5, 0.5);
static const int4 v_i32_one = int4(1, 1, 1, 1);

float4 builtins()
{
    int s1_ = (true ? 1 : 0);
    float4 s2_ = (true ? v_f32_one : v_f32_zero);
    float4 s3_ = (bool4(false, false, false, false) ? v_f32_zero : v_f32_one);
    float4 m1_ = lerp(v_f32_zero, v_f32_one, v_f32_half);
    float4 m2_ = lerp(v_f32_zero, v_f32_one, 0.1);
    float b1_ = asfloat(v_i32_one.x);
    float4 b2_ = asfloat(v_i32_one);
    int4 v_i32_zero = int4(v_f32_zero);
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
    float2x2 m0_ = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    float4x4 m1_1 = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    uint2 cit0_ = (0u).xx;
    float2x2 cit1_ = float2x2((0.0).xx, (0.0).xx);
    int cit2_[4] = Constructarray4_int_(0, 1, 2, 3);
    bool ic0_ = bool((bool)0);
    int ic1_ = int((int)0);
    uint ic2_ = uint((uint)0);
    float ic3_ = float((float)0);
    uint2 ic4_ = uint2((uint2)0);
    float2x3 ic5_ = float2x3((float2x3)0);
    uint2 ic6_ = asuint((uint2)0);
    float2x3 ic7_ = asfloat((float2x3)0);
    float _expr71 = foo.a.x;
    return _expr71;
}

void logical()
{
    bool neg0_ = !(true);
    bool2 neg1_ = !((true).xx);
    bool or_ = (true || false);
    bool and_ = (true && false);
    bool bitwise_or0_ = (true | false);
    bool3 bitwise_or1_ = ((true).xxx | (false).xxx);
    bool bitwise_and0_ = (true & false);
    bool4 bitwise_and1_ = ((true).xxxx & (false).xxxx);
}

void arithmetic()
{
    int2 neg1_1 = -((1).xx);
    float2 neg2_ = -((1.0).xx);
    int add0_ = (2 + 1);
    uint add1_ = (2u + 1u);
    float add2_ = (2.0 + 1.0);
    int2 add3_ = ((2).xx + (1).xx);
    uint3 add4_ = ((2u).xxx + (1u).xxx);
    float4 add5_ = ((2.0).xxxx + (1.0).xxxx);
    int sub0_ = (2 - 1);
    uint sub1_ = (2u - 1u);
    float sub2_ = (2.0 - 1.0);
    int2 sub3_ = ((2).xx - (1).xx);
    uint3 sub4_ = ((2u).xxx - (1u).xxx);
    float4 sub5_ = ((2.0).xxxx - (1.0).xxxx);
    int mul0_ = (2 * 1);
    uint mul1_ = (2u * 1u);
    float mul2_ = (2.0 * 1.0);
    int2 mul3_ = ((2).xx * (1).xx);
    uint3 mul4_ = ((2u).xxx * (1u).xxx);
    float4 mul5_ = ((2.0).xxxx * (1.0).xxxx);
    int div0_ = (2 / 1);
    uint div1_ = (2u / 1u);
    float div2_ = (2.0 / 1.0);
    int2 div3_ = ((2).xx / (1).xx);
    uint3 div4_ = ((2u).xxx / (1u).xxx);
    float4 div5_ = ((2.0).xxxx / (1.0).xxxx);
    int rem0_ = (2 % 1);
    uint rem1_ = (2u % 1u);
    float rem2_ = fmod(2.0, 1.0);
    int2 rem3_ = ((2).xx % (1).xx);
    uint3 rem4_ = ((2u).xxx % (1u).xxx);
    float4 rem5_ = fmod((2.0).xxxx, (1.0).xxxx);
    {
        int2 add0_1 = ((2).xx + (1).xx);
        int2 add1_1 = ((2).xx + (1).xx);
        uint2 add2_1 = ((2u).xx + (1u).xx);
        uint2 add3_1 = ((2u).xx + (1u).xx);
        float2 add4_1 = ((2.0).xx + (1.0).xx);
        float2 add5_1 = ((2.0).xx + (1.0).xx);
        int2 sub0_1 = ((2).xx - (1).xx);
        int2 sub1_1 = ((2).xx - (1).xx);
        uint2 sub2_1 = ((2u).xx - (1u).xx);
        uint2 sub3_1 = ((2u).xx - (1u).xx);
        float2 sub4_1 = ((2.0).xx - (1.0).xx);
        float2 sub5_1 = ((2.0).xx - (1.0).xx);
        int2 mul0_1 = ((2).xx * 1);
        int2 mul1_1 = (2 * (1).xx);
        uint2 mul2_1 = ((2u).xx * 1u);
        uint2 mul3_1 = (2u * (1u).xx);
        float2 mul4_1 = ((2.0).xx * 1.0);
        float2 mul5_1 = (2.0 * (1.0).xx);
        int2 div0_1 = ((2).xx / (1).xx);
        int2 div1_1 = ((2).xx / (1).xx);
        uint2 div2_1 = ((2u).xx / (1u).xx);
        uint2 div3_1 = ((2u).xx / (1u).xx);
        float2 div4_1 = ((2.0).xx / (1.0).xx);
        float2 div5_1 = ((2.0).xx / (1.0).xx);
        int2 rem0_1 = ((2).xx % (1).xx);
        int2 rem1_1 = ((2).xx % (1).xx);
        uint2 rem2_1 = ((2u).xx % (1u).xx);
        uint2 rem3_1 = ((2u).xx % (1u).xx);
        float2 rem4_1 = fmod((2.0).xx, (1.0).xx);
        float2 rem5_1 = fmod((2.0).xx, (1.0).xx);
    }
    float3x3 add = ((float3x3)0 + (float3x3)0);
    float3x3 sub = ((float3x3)0 - (float3x3)0);
    float3x3 mul_scalar0_ = mul(1.0, (float3x3)0);
    float3x3 mul_scalar1_ = mul((float3x3)0, 2.0);
    float3 mul_vector0_ = mul((1.0).xxxx, (float4x3)0);
    float4 mul_vector1_ = mul((float4x3)0, (2.0).xxx);
    float3x3 mul_ = mul((float3x4)0, (float4x3)0);
}

void bit()
{
    int flip0_ = ~(1);
    uint flip1_ = ~(1u);
    int2 flip2_ = ~((1).xx);
    uint3 flip3_ = ~((1u).xxx);
    int or0_ = (2 | 1);
    uint or1_ = (2u | 1u);
    int2 or2_ = ((2).xx | (1).xx);
    uint3 or3_ = ((2u).xxx | (1u).xxx);
    int and0_ = (2 & 1);
    uint and1_ = (2u & 1u);
    int2 and2_ = ((2).xx & (1).xx);
    uint3 and3_ = ((2u).xxx & (1u).xxx);
    int xor0_ = (2 ^ 1);
    uint xor1_ = (2u ^ 1u);
    int2 xor2_ = ((2).xx ^ (1).xx);
    uint3 xor3_ = ((2u).xxx ^ (1u).xxx);
    int shl0_ = (2 << 1u);
    uint shl1_ = (2u << 1u);
    int2 shl2_ = ((2).xx << (1u).xx);
    uint3 shl3_ = ((2u).xxx << (1u).xxx);
    int shr0_ = (2 >> 1u);
    uint shr1_ = (2u >> 1u);
    int2 shr2_ = ((2).xx >> (1u).xx);
    uint3 shr3_ = ((2u).xxx >> (1u).xxx);
}

void comparison()
{
    bool eq0_ = (2 == 1);
    bool eq1_ = (2u == 1u);
    bool eq2_ = (2.0 == 1.0);
    bool2 eq3_ = ((2).xx == (1).xx);
    bool3 eq4_ = ((2u).xxx == (1u).xxx);
    bool4 eq5_ = ((2.0).xxxx == (1.0).xxxx);
    bool neq0_ = (2 != 1);
    bool neq1_ = (2u != 1u);
    bool neq2_ = (2.0 != 1.0);
    bool2 neq3_ = ((2).xx != (1).xx);
    bool3 neq4_ = ((2u).xxx != (1u).xxx);
    bool4 neq5_ = ((2.0).xxxx != (1.0).xxxx);
    bool lt0_ = (2 < 1);
    bool lt1_ = (2u < 1u);
    bool lt2_ = (2.0 < 1.0);
    bool2 lt3_ = ((2).xx < (1).xx);
    bool3 lt4_ = ((2u).xxx < (1u).xxx);
    bool4 lt5_ = ((2.0).xxxx < (1.0).xxxx);
    bool lte0_ = (2 <= 1);
    bool lte1_ = (2u <= 1u);
    bool lte2_ = (2.0 <= 1.0);
    bool2 lte3_ = ((2).xx <= (1).xx);
    bool3 lte4_ = ((2u).xxx <= (1u).xxx);
    bool4 lte5_ = ((2.0).xxxx <= (1.0).xxxx);
    bool gt0_ = (2 > 1);
    bool gt1_ = (2u > 1u);
    bool gt2_ = (2.0 > 1.0);
    bool2 gt3_ = ((2).xx > (1).xx);
    bool3 gt4_ = ((2u).xxx > (1u).xxx);
    bool4 gt5_ = ((2.0).xxxx > (1.0).xxxx);
    bool gte0_ = (2 >= 1);
    bool gte1_ = (2u >= 1u);
    bool gte2_ = (2.0 >= 1.0);
    bool2 gte3_ = ((2).xx >= (1).xx);
    bool3 gte4_ = ((2u).xxx >= (1u).xxx);
    bool4 gte5_ = ((2.0).xxxx >= (1.0).xxxx);
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
    int _expr33 = a_1;
    a_1 = (_expr33 + 1);
    int _expr36 = a_1;
    a_1 = (_expr36 - 1);
    vec0_ = (int3)0;
    int _expr43 = vec0_.y;
    vec0_.y = (_expr43 + 1);
    int _expr48 = vec0_.y;
    vec0_.y = (_expr48 - 1);
    return;
}

void negation_avoids_prefix_decrement()
{
    int p1_ = -(-2);
    int p2_ = -(-3);
    int p3_ = -(-(4));
    int p4_ = -(-(-5));
    int p5_ = -(-(-(-(6))));
    int p6_ = -(-(-(-(-7))));
    int p7_ = -(-(-(-(-8))));
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e0 = builtins();
    const float4 _e1 = splat();
    const float3 _e4 = bool_cast(v_f32_one.xyz);
    const float _e5 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
