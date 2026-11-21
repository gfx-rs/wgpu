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

    s.m = float3x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0));
    float _e17 = s.m[0].x;
    out_.Store(0, asuint(_e17));
    float _e23 = s.m[2].y;
    out_.Store(4, asuint(_e23));
    return;
}
