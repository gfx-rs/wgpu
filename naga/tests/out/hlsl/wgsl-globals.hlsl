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

float3 ZeroValuefloat3() {
    return (float3)0;
}

FooStruct ConstructFooStruct(float3 arg0, float arg1) {
    FooStruct ret = (FooStruct)0;
    ret.v3_ = arg0;
    ret.v1_ = arg1;
    return ret;
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
    float Foo = 1.0;
    bool at = true;
    int idx = (int)0;
    float3 arg = (float3)0;
    uint zero_init_index = (uint)0;

    zero_init_index = local_invocation_index;
    while(true) {
        uint _e2 = zero_init_index;
        if ((_e2 >= 10u)) {
            break;
        }
        wg[min(uint(_e2), 9u)] = 0.0;
        zero_init_index = (_e2 + 1u);
    }
    if ((local_invocation_index == 0u)) {
        at_1 = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    idx = int(1);
    arg = ZeroValuefloat3();
    alignment.Store3(0, asuint((1.0).xxx));
    alignment.Store(0+0, asuint(1.0));
    alignment.Store(0+0, asuint(2.0));
    int _e24 = idx;
    alignment.Store(_e24*4+0, asuint(3.0));
    FooStruct _e27 = ConstructFooStruct(asfloat(alignment.Load3(0)), asfloat(alignment.Load(12)));
    arg = _e27.v3_;
    float2 _e35 = global_nested_arrays_of_matrices_4x2_[0][0].c_0_;
    float2 _e37 = global_nested_arrays_of_matrices_4x2_[0][0].c_1_;
    float2 _e39 = global_nested_arrays_of_matrices_4x2_[0][0].c_2_;
    float2 _e41 = global_nested_arrays_of_matrices_4x2_[0][0].c_3_;
    float4 _e47 = global_nested_arrays_of_matrices_2x4_[0][0][0];
    wg[7] = mul(_e47, float4x2(_e35, _e37, _e39, _e41)).x;
    float2 _e54 = global_mat.c_0_;
    float2 _e56 = global_mat.c_1_;
    float2 _e58 = global_mat.c_2_;
    float3 _e61 = global_vec;
    wg[6] = mul(_e61, float3x2(_e54, _e56, _e58)).x;
    float _e69 = asfloat(dummy.Load(4+8));
    wg[5] = _e69;
    float _e75 = float_vecs[0].w;
    wg[4] = _e75;
    float _e80 = asfloat(alignment.Load(12));
    wg[3] = _e80;
    float _e86 = asfloat(alignment.Load(0+0));
    wg[2] = _e86;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    { uint dummy_1 = 0; InterlockedExchange(at_1, 2u, dummy_1); }
    return;
}
