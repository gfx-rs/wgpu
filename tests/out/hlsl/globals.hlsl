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
cbuffer global_vec : register(b4) { float4 global_vec; }
cbuffer global_mat : register(b5) { row_major float4x4 global_mat; }

void test_msl_packed_vec3_as_arg(float3 arg)
{
    return;
}

void test_msl_packed_vec3_()
{
    int idx = 1;

    alignment.Store3(0, asuint((1.0).xxx));
    alignment.Store(0+0, asuint(1.0));
    alignment.Store(0+0, asuint(2.0));
    int _expr21 = idx;
    alignment.Store(_expr21*4+0, asuint(3.0));
    Foo data = {asfloat(alignment.Load3(0)), asfloat(alignment.Load(12))};
    float3 unnamed = data.v3_;
    float2 unnamed_1 = data.v3_.zx;
    test_msl_packed_vec3_as_arg(data.v3_);
    float3 unnamed_2 = mul(float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)), data.v3_);
    float3 unnamed_3 = mul(data.v3_, float3x3(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0)));
    float3 unnamed_4 = (data.v3_ * 2.0);
    float3 unnamed_5 = (2.0 * data.v3_);
}

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

    test_msl_packed_vec3_();
    float4x4 _expr10 = global_mat;
    float4 _expr11 = global_vec;
    wg[6] = mul(_expr11, _expr10).x;
    float _expr19 = asfloat(dummy.Load(4+8));
    wg[5] = _expr19;
    float _expr25 = float_vecs[0].w;
    wg[4] = _expr25;
    float _expr29 = asfloat(alignment.Load(12));
    wg[3] = _expr29;
    float _expr34 = asfloat(alignment.Load(0+0));
    wg[2] = _expr34;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    at_1 = 2u;
    return;
}
