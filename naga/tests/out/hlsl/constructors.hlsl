struct Foo {
    float4 a;
    int b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

typedef float2x2 ret_Constructarray1_float2x2_[1];
ret_Constructarray1_float2x2_ Constructarray1_float2x2_(float2x2 arg0) {
    float2x2 ret[1] = { arg0 };
    return ret;
}

typedef int ret_Constructarray4_int_[4];
ret_Constructarray4_int_ Constructarray4_int_(int arg0, int arg1, int arg2, int arg3) {
    int ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

bool ZeroValuebool() {
    return (bool)0;
}

int ZeroValueint() {
    return (int)0;
}

uint ZeroValueuint() {
    return (uint)0;
}

float ZeroValuefloat() {
    return (float)0;
}

uint2 ZeroValueuint2() {
    return (uint2)0;
}

float2x2 ZeroValuefloat2x2() {
    return (float2x2)0;
}

typedef Foo ret_ZeroValuearray3_Foo_[3];
ret_ZeroValuearray3_Foo_ ZeroValuearray3_Foo_() {
    return (Foo[3])0;
}

Foo ZeroValueFoo() {
    return (Foo)0;
}

static const float3 const2_ = float3(0.0, 1.0, 2.0);
static const float2x2 const3_ = float2x2(float2(0.0, 1.0), float2(2.0, 3.0));
static const float2x2 const4_[1] = Constructarray1_float2x2_(float2x2(float2(0.0, 1.0), float2(2.0, 3.0)));
static const bool cz0_ = ZeroValuebool();
static const int cz1_ = ZeroValueint();
static const uint cz2_ = ZeroValueuint();
static const float cz3_ = ZeroValuefloat();
static const uint2 cz4_ = ZeroValueuint2();
static const float2x2 cz5_ = ZeroValuefloat2x2();
static const Foo cz6_[3] = ZeroValuearray3_Foo_();
static const Foo cz7_ = ZeroValueFoo();
static const int cp3_[4] = Constructarray4_int_(0, 1, 2, 3);

Foo ConstructFoo(float4 arg0, int arg1) {
    Foo ret = (Foo)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

float2x3 ZeroValuefloat2x3() {
    return (float2x3)0;
}

[numthreads(1, 1, 1)]
void main()
{
    Foo foo = (Foo)0;

    foo = ConstructFoo((1.0).xxxx, 1);
    float2x2 m0_ = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    float4x4 m1_ = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    uint2 cit0_ = (0u).xx;
    float2x2 cit1_ = float2x2((0.0).xx, (0.0).xx);
    int cit2_[4] = Constructarray4_int_(0, 1, 2, 3);
    uint2 ic4_ = uint2(0u, 0u);
    float2x3 ic5_ = float2x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0));
}
