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
    Foo ret;
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
    return (((((float4((int4(s1_.xxxx) + v_i32_zero)) + s2_) + m1_) + m2_) + float4(b1_.xxxx)) + b2_);
}

float4 splat()
{
    float2 a_1 = (((float2(1.0.xx) + float2(2.0.xx)) - float2(3.0.xx)) / float2(4.0.xx));
    int4 b = (int4(5.xxxx) % int4(2.xxxx));
    return (a_1.xyxy + float4(b));
}

int unary()
{
    if (!true) {
        return 1;
    } else {
        return ~1;
    }
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
    bool unnamed = false;
    int unnamed_1 = 0;
    uint unnamed_2 = 0u;
    float unnamed_3 = 0.0;
    uint2 unnamed_4 = uint2(0u, 0u);
    float2x2 unnamed_5 = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));
    Foo unnamed_6[3] = Constructarray3_Foo_(ConstructFoo(float4(0.0, 0.0, 0.0, 0.0), 0), ConstructFoo(float4(0.0, 0.0, 0.0, 0.0), 0), ConstructFoo(float4(0.0, 0.0, 0.0, 0.0), 0));
    Foo unnamed_7 = ConstructFoo(float4(0.0, 0.0, 0.0, 0.0), 0);
    uint2 unnamed_8 = (uint2)0;
    float2x2 unnamed_9 = (float2x2)0;
    int unnamed_10[4] = {(int)0,(int)0,(int)0,(int)0};

    foo = ConstructFoo(float4(1.0.xxxx), 1);
    float2x2 mat2comp = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    float4x4 mat4comp = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    unnamed_8 = uint2(0u.xx);
    unnamed_9 = float2x2(float2(0.0.xx), float2(0.0.xx));
    {
        int _result[4]=Constructarray4_int_(0, 1, 2, 3);
        for(int _i=0; _i<4; ++_i) unnamed_10[_i] = _result[_i];
    }
    float _expr70 = foo.a.x;
    return _expr70;
}

void modulo()
{
    int a_2 = (1 % 1);
    float b_1 = (1.0 % 1.0);
    int3 c = (int3(1.xxx) % int3(1.xxx));
    float3 d = (float3(1.0.xxx) % float3(1.0.xxx));
}

void scalar_times_matrix()
{
    float4x4 model = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    float4x4 assertion = mul(model, 2.0);
}

void logical()
{
    bool a_3 = (true | false);
    bool b_2 = (true & false);
}

void binary_assignment()
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
    a = (_expr21 ^ 0);
    int _expr24 = a;
    a = (_expr24 & 0);
    int _expr27 = a;
    a = (_expr27 + 1);
    int _expr30 = a;
    a = (_expr30 - 1);
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e4 = builtins();
    const float4 _e5 = splat();
    const int _e6 = unary();
    const float3 _e8 = bool_cast(float4(1.0, 1.0, 1.0, 1.0).xyz);
    const float _e9 = constructors();
    modulo();
    scalar_times_matrix();
    logical();
    binary_assignment();
    return;
}
