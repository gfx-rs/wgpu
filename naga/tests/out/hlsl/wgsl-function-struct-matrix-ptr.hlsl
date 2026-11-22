struct S {
    float a;
    int _pad1_0;
    row_major float3x2 m;
};

RWByteAddressBuffer out_ : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    S s = (S)0;
    float local = (float)0;

    s.m = float3x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0));
    s.m[1] = float2(5.0, 6.0);
    float _e22 = s.m[1].y;
    out_.Store(0, asuint(_e22));
    float _e28 = s.m[2].y;
    local = _e28;
    float _e30 = local;
    out_.Store(4, asuint(_e30));
    return;
}
