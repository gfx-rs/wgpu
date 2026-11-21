struct S {
    float a;
    int _pad1_0;
    row_major float3x2 m;
};

RWByteAddressBuffer out_ : register(u0);

void store_column(inout float3x2 p)
{
    p[1] = float2(5.0, 6.0);
    return;
}

float read_matrix(inout float3x2 p_1)
{
    float _e3 = p_1[2].y;
    return _e3;
}

[numthreads(1, 1, 1)]
void main()
{
    S s = (S)0;

    s.m = float3x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0));
    store_column(s.m);
    float _e18 = s.m[1].y;
    out_.Store(0, asuint(_e18));
    const float _e22 = read_matrix(s.m);
    out_.Store(4, asuint(_e22));
    return;
}
