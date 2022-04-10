static const bool Foo_2 = true;

struct Foo {
    float3 v3_;
    float v1_;
};

groupshared float wg[10];
groupshared uint at_1;
RWByteAddressBuffer alignment : register(u1);
ByteAddressBuffer dummy : register(t2);
cbuffer float_vecs : register(b3) { float4 float_vecs[20]; }

uint NagaBufferLength(ByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    float3 unnamed = (float3)0;
    float2 unnamed_1 = (float2)0;
    int idx = 1;
    float Foo_1 = 1.0;
    bool at = true;

    float _expr9 = asfloat(alignment.Load(12));
    wg[3] = _expr9;
    float _expr14 = asfloat(alignment.Load(0+0));
    wg[2] = _expr14;
    float3 _expr16 = asfloat(alignment.Load3(0));
    unnamed = _expr16;
    float3 _expr19 = asfloat(alignment.Load3(0));
    unnamed_1 = _expr19.zx;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    at_1 = 2u;
    alignment.Store3(0, asuint(float3(1.0.xxx)));
    alignment.Store(0+0, asuint(1.0));
    alignment.Store(0+0, asuint(2.0));
    int _expr42 = idx;
    alignment.Store(_expr42*4+0, asuint(3.0));
    float3 _expr47 = asfloat(alignment.Load3(0));
    float3 unnamed_2 = mul(float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)), _expr47);
    float3 _expr50 = asfloat(alignment.Load3(0));
    float3 unnamed_3 = mul(_expr50, float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
}
