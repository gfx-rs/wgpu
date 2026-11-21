struct FooStruct {
    float3 v3_;
    float v1_;
};

struct mat3x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
    float2 c_2_;
};

struct mat4x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
    float2 c_2_;
    float2 c_3_;
};

static const bool Foo_1 = true;

groupshared float wg[10];
groupshared uint at_1;
RWByteAddressBuffer alignment : register(u1);
ByteAddressBuffer dummy : register(t2);
cbuffer float_vecs : register(b3) { float4 float_vecs[20]; }
cbuffer global_vec : register(b4) { float3 global_vec; }
cbuffer global_mat : register(b5) { mat3x2f32_std140_ global_mat; }
cbuffer global_nested_arrays_of_matrices_2x4_ : register(b6) { row_major float2x4 global_nested_arrays_of_matrices_2x4_[2][2]; }
cbuffer global_nested_arrays_of_matrices_4x2_ : register(b7) { mat4x2f32_std140_ global_nested_arrays_of_matrices_4x2_[2][2]; }

void test_msl_packed_vec3_as_arg(float3 arg)
{
    return;
}

float3x3 ZeroValuefloat3x3() {
    return (float3x3)0;
}

FooStruct ConstructFooStruct(float3 arg0, float arg1) {
    FooStruct ret = (FooStruct)0;
    ret.v3_ = arg0;
    ret.v1_ = arg1;
    return ret;
}

void test_msl_packed_vec3_()
{
    int idx = int(1);

    alignment.Store3(0, asuint((1.0).xxx));
    alignment.Store(0+0, asuint(1.0));
    alignment.Store(0+0, asuint(2.0));
    int _e16 = idx;
    alignment.Store(_e16*4+0, asuint(3.0));
    FooStruct data = ConstructFooStruct(asfloat(alignment.Load3(0)), asfloat(alignment.Load(12)));
    float3 l0_ = data.v3_;
    float2 l1_ = data.v3_.zx;
    test_msl_packed_vec3_as_arg(data.v3_);
    float3 mvm0_ = mul(ZeroValuefloat3x3(), data.v3_);
    float3 mvm1_ = mul(data.v3_, ZeroValuefloat3x3());
    float3 svm0_ = (data.v3_ * 2.0);
    float3 svm1_ = (2.0 * data.v3_);
    return;
}

uint NagaBufferLength(ByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

[numthreads(1, 1, 1)]
void main(uint local_invocation_index : SV_GroupIndex)
{
    [loop]
    for (uint zero_flat_index = local_invocation_index; zero_flat_index < 10u; zero_flat_index += 1u) {
        uint zero_index = zero_flat_index;
        wg[zero_index] = (float)0;
    }
    if (local_invocation_index == 0) {
        at_1 = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    float Foo = 1.0;
    bool at = true;

    test_msl_packed_vec3_();
    float2 _e6 = global_nested_arrays_of_matrices_4x2_[0][0].c_0_;
    float2 _e8 = global_nested_arrays_of_matrices_4x2_[0][0].c_1_;
    float2 _e10 = global_nested_arrays_of_matrices_4x2_[0][0].c_2_;
    float2 _e12 = global_nested_arrays_of_matrices_4x2_[0][0].c_3_;
    float4 _e18 = global_nested_arrays_of_matrices_2x4_[0][0][0];
    wg[7] = mul(_e18, float4x2(_e6, _e8, _e10, _e12)).x;
    float2 _e25 = global_mat.c_0_;
    float2 _e27 = global_mat.c_1_;
    float2 _e29 = global_mat.c_2_;
    float3 _e32 = global_vec;
    wg[6] = mul(_e32, float3x2(_e25, _e27, _e29)).x;
    float _e40 = asfloat(dummy.Load(4+8));
    wg[5] = _e40;
    float _e46 = float_vecs[0].w;
    wg[4] = _e46;
    float _e51 = asfloat(alignment.Load(12));
    wg[3] = _e51;
    float _e57 = asfloat(alignment.Load(0+0));
    wg[2] = _e57;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    { uint dummy_1 = 0; InterlockedExchange(at_1, 2u, dummy_1); }
    return;
}
