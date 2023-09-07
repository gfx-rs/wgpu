typedef struct { float2 _0; float2 _1; float2 _2; } __mat3x2;
float2 __get_col_of_mat3x2(__mat3x2 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat3x2(__mat3x2 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    case 2: { mat._2 = value; break; }
    }
}
void __set_el_of_mat3x2(__mat3x2 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    case 2: { mat._2[vec_idx] = value; break; }
    }
}

typedef struct { float2 _0; float2 _1; float2 _2; float2 _3; } __mat4x2;
float2 __get_col_of_mat4x2(__mat4x2 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    case 3: { return mat._3; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat4x2(__mat4x2 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    case 2: { mat._2 = value; break; }
    case 3: { mat._3 = value; break; }
    }
}
void __set_el_of_mat4x2(__mat4x2 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    case 2: { mat._2[vec_idx] = value; break; }
    case 3: { mat._3[vec_idx] = value; break; }
    }
}

struct FooStruct {
    float3 v3_;
    float v1_;
};

static const bool Foo_1 = true;

groupshared float wg[10];
groupshared uint at_1;
RWByteAddressBuffer alignment : register(u1);
ByteAddressBuffer dummy : register(t2);
cbuffer float_vecs : register(b3) { float4 float_vecs[20]; }
cbuffer global_vec : register(b4) { float3 global_vec; }
cbuffer global_mat : register(b5) { __mat3x2 global_mat; }
cbuffer global_nested_arrays_of_matrices_2x4_ : register(b6) { row_major float2x4 global_nested_arrays_of_matrices_2x4_[2][2]; }
cbuffer global_nested_arrays_of_matrices_4x2_ : register(b7) { __mat4x2 global_nested_arrays_of_matrices_4x2_[2][2]; }

void test_msl_packed_vec3_as_arg(float3 arg)
{
    return;
}

FooStruct ConstructFooStruct(float3 arg0, float arg1) {
    FooStruct ret = (FooStruct)0;
    ret.v3_ = arg0;
    ret.v1_ = arg1;
    return ret;
}

void test_msl_packed_vec3_()
{
    int idx = (int)0;

    alignment.Store3(0, asuint((1.0).xxx));
    idx = 1;
    alignment.Store(0+0, asuint(1.0));
    alignment.Store(0+0, asuint(2.0));
    int _expr16 = idx;
    alignment.Store(_expr16*4+0, asuint(3.0));
    FooStruct data = ConstructFooStruct(asfloat(alignment.Load3(0)), asfloat(alignment.Load(12)));
    float3 l0_ = data.v3_;
    float2 l1_ = data.v3_.zx;
    test_msl_packed_vec3_as_arg(data.v3_);
    float3 mvm0_ = mul((float3x3)0, data.v3_);
    float3 mvm1_ = mul(data.v3_, (float3x3)0);
    float3 svm0_ = (data.v3_ * 2.0);
    float3 svm1_ = (2.0 * data.v3_);
}

uint NagaBufferLength(ByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

[numthreads(1, 1, 1)]
void main(uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
        wg = (float[10])0;
        at_1 = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    float Foo = (float)0;
    bool at = (bool)0;

    test_msl_packed_vec3_();
    float4x2 _expr5 = ((float4x2)global_nested_arrays_of_matrices_4x2_[0][0]);
    float4 _expr10 = global_nested_arrays_of_matrices_2x4_[0][0][0];
    wg[7] = mul(_expr10, _expr5).x;
    float3x2 _expr16 = ((float3x2)global_mat);
    float3 _expr18 = global_vec;
    wg[6] = mul(_expr18, _expr16).x;
    float _expr26 = asfloat(dummy.Load(4+8));
    wg[5] = _expr26;
    float _expr32 = float_vecs[0].w;
    wg[4] = _expr32;
    float _expr37 = asfloat(alignment.Load(12));
    wg[3] = _expr37;
    float _expr43 = asfloat(alignment.Load(0+0));
    wg[2] = _expr43;
    alignment.Store(12, asuint(4.0));
    wg[1] = float(((NagaBufferLength(dummy) - 0) / 8));
    at_1 = 2u;
    Foo = 1.0;
    at = true;
    return;
}
