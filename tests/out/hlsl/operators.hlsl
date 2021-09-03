static const float4 v_f32_one = float4(1.0, 1.0, 1.0, 1.0);
static const float4 v_f32_zero = float4(0.0, 0.0, 0.0, 0.0);
static const float4 v_f32_half = float4(0.5, 0.5, 0.5, 0.5);

struct Foo {
    float4 a;
    int b;
};

float4 builtins()
{
    int s1_ = (true ? 1 : 0);
    float4 s2_ = (true ? float4(1.0, 1.0, 1.0, 1.0) : float4(0.0, 0.0, 0.0, 0.0));
    float4 s3_ = (bool4(false, false, false, false) ? float4(0.0, 0.0, 0.0, 0.0) : float4(1.0, 1.0, 1.0, 1.0));
    float4 m1_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), float4(0.5, 0.5, 0.5, 0.5));
    float4 m2_ = lerp(float4(0.0, 0.0, 0.0, 0.0), float4(1.0, 1.0, 1.0, 1.0), 0.1);
    return (((float4(int4(s1_.xxxx)) + s2_) + m1_) + m2_);
}

float4 splat()
{
    float2 a = (((float2(1.0.xx) + float2(2.0.xx)) - float2(3.0.xx)) / float2(4.0.xx));
    int4 b = (int4(5.xxxx) % int4(2.xxxx));
    return (a.xyxy + float4(b));
}

int unary()
{
    if (!true) {
        return 1;
    } else {
        return !1;
    }
}

Foo ConstructFoo(float4 arg0, int arg1) {
    Foo ret;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

float constructors()
{
    Foo foo = (Foo)0;

    foo = ConstructFoo(float4(1.0.xxxx), 1);
    float4 _expr9 = foo.a;
    return _expr9.x;
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e3 = builtins();
    const float4 _e4 = splat();
    const int _e5 = unary();
    const float _e6 = constructors();
    return;
}
