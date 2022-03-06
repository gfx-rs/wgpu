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
    float Foo_1 = 1.0;
    bool at = true;

    float _expr9 = asfloat(alignment.Load(12));
    wg[3] = _expr9;
    float _expr14 = asfloat(alignment.Load(0+0));
    wg[2] = _expr14;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    at_1 = 2u;
    return;
}
