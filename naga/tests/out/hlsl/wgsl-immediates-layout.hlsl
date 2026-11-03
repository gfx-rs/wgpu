struct Inner {
    uint a;
    uint b;
};

struct NestedAfterScalar {
    uint x;
    Inner inner;
    uint y;
};

struct NestedAfterVec3_ {
    uint3 v;
    Inner inner;
    uint y;
    int _end_pad_0;
    int _end_pad_1;
};

struct Mixed {
    int s;
    int _pad1_0;
    float2 m_0; float2 m_1; float2 m_2;
    Inner inner;
    int _pad3_0;
    int _pad3_1;
    row_major float4x3 m4_;
    int _pad4_0;
    int4 v4_;
    float f;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

cbuffer nested_after_scalar_block : register(b0) { uint4 nested_after_scalar_raw[1]; };
static NestedAfterScalar nested_after_scalar;
cbuffer nested_after_vec3_block : register(b0) { uint4 nested_after_vec3_raw[2]; };
static NestedAfterVec3_ nested_after_vec3_;
cbuffer mixed_block : register(b0) { uint4 mixed_raw[9]; };
static Mixed mixed;
cbuffer non_struct_block : register(b0) { uint4 non_struct_raw[1]; };
static float4 non_struct;
RWByteAddressBuffer out_ : register(u0);
Inner ConstructInner(uint arg0, uint arg1) {
    Inner ret = (Inner)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

NestedAfterScalar ConstructNestedAfterScalar(uint arg0, Inner arg1, uint arg2) {
    NestedAfterScalar ret = (NestedAfterScalar)0;
    ret.x = arg0;
    ret.inner = arg1;
    ret.y = arg2;
    return ret;
}

NestedAfterVec3_ ConstructNestedAfterVec3_(uint3 arg0, Inner arg1, uint arg2) {
    NestedAfterVec3_ ret = (NestedAfterVec3_)0;
    ret.v = arg0;
    ret.inner = arg1;
    ret.y = arg2;
    return ret;
}

Mixed ConstructMixed(int arg0, float3x2 arg1, Inner arg2, float4x3 arg3, int4 arg4, float arg5) {
    Mixed ret = (Mixed)0;
    ret.s = arg0;
    ret.m_0 = arg1[0];
    ret.m_1 = arg1[1];
    ret.m_2 = arg1[2];
    ret.inner = arg2;
    ret.m4_ = arg3;
    ret.v4_ = arg4;
    ret.f = arg5;
    return ret;
}


uint read_nested_after_vec3_()
{
    uint _e3 = nested_after_vec3_.inner.b;
    return _e3;
}

[numthreads(1, 1, 1)]
void scalar_then_struct()
{
    nested_after_scalar = ConstructNestedAfterScalar(asuint(nested_after_scalar_raw[0].x), ConstructInner(asuint(nested_after_scalar_raw[0].y), asuint(nested_after_scalar_raw[0].z)), asuint(nested_after_scalar_raw[0].w));
    uint _e4 = nested_after_scalar.x;
    out_.Store(0, asuint(_e4));
    uint _e10 = nested_after_scalar.inner.a;
    out_.Store(4, asuint(_e10));
    uint _e16 = nested_after_scalar.inner.b;
    out_.Store(8, asuint(_e16));
    uint _e21 = nested_after_scalar.y;
    out_.Store(12, asuint(_e21));
    return;
}

[numthreads(1, 1, 1)]
void vec3_then_struct()
{
    nested_after_vec3_ = ConstructNestedAfterVec3_(uint3(asuint(nested_after_vec3_raw[0].x), asuint(nested_after_vec3_raw[0].y), asuint(nested_after_vec3_raw[0].z)), ConstructInner(asuint(nested_after_vec3_raw[0].w), asuint(nested_after_vec3_raw[1].x)), asuint(nested_after_vec3_raw[1].y));
    uint _e5 = nested_after_vec3_.v.z;
    out_.Store(0, asuint(_e5));
    uint _e11 = nested_after_vec3_.inner.a;
    out_.Store(4, asuint(_e11));
    const uint _e14 = read_nested_after_vec3_();
    out_.Store(8, asuint(_e14));
    uint _e19 = nested_after_vec3_.y;
    out_.Store(12, asuint(_e19));
    return;
}

float3x2 GetMatmOnMixed(Mixed obj) {
    return float3x2(obj.m_0, obj.m_1, obj.m_2);
}

void SetMatmOnMixed(Mixed obj, float3x2 mat) {
    obj.m_0 = mat[0];
    obj.m_1 = mat[1];
    obj.m_2 = mat[2];
}

void SetMatVecmOnMixed(Mixed obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0 = vec; break; }
    case 1: { obj.m_1 = vec; break; }
    case 2: { obj.m_2 = vec; break; }
    }
}

void SetMatScalarmOnMixed(Mixed obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0[vec_idx] = scalar; break; }
    case 1: { obj.m_1[vec_idx] = scalar; break; }
    case 2: { obj.m_2[vec_idx] = scalar; break; }
    }
}

[numthreads(1, 1, 1)]
void matrices_and_vectors()
{
    mixed = ConstructMixed(asint(mixed_raw[0].x), float3x2(float2(asfloat(mixed_raw[0].z), asfloat(mixed_raw[0].w)), float2(asfloat(mixed_raw[1].x), asfloat(mixed_raw[1].y)), float2(asfloat(mixed_raw[1].z), asfloat(mixed_raw[1].w))), ConstructInner(asuint(mixed_raw[2].x), asuint(mixed_raw[2].y)), float4x3(float3(asfloat(mixed_raw[3].x), asfloat(mixed_raw[3].y), asfloat(mixed_raw[3].z)), float3(asfloat(mixed_raw[4].x), asfloat(mixed_raw[4].y), asfloat(mixed_raw[4].z)), float3(asfloat(mixed_raw[5].x), asfloat(mixed_raw[5].y), asfloat(mixed_raw[5].z)), float3(asfloat(mixed_raw[6].x), asfloat(mixed_raw[6].y), asfloat(mixed_raw[6].z))), int4(asint(mixed_raw[7].x), asint(mixed_raw[7].y), asint(mixed_raw[7].z), asint(mixed_raw[7].w)), asfloat(mixed_raw[8].x));
    int _e4 = mixed.s;
    out_.Store(0, asuint(asuint(_e4)));
    float _e12 = GetMatmOnMixed(mixed)[1].y;
    out_.Store(4, asuint(asuint(_e12)));
    uint _e19 = mixed.inner.b;
    out_.Store(8, asuint(_e19));
    float _e26 = mixed.m4_[2].z;
    out_.Store(12, asuint(asuint(_e26)));
    int _e33 = mixed.v4_.w;
    out_.Store(16, asuint(asuint(_e33)));
    float _e39 = mixed.f;
    out_.Store(20, asuint(asuint(_e39)));
    return;
}

[numthreads(1, 1, 1)]
void whole_vector()
{
    non_struct = float4(asfloat(non_struct_raw[0].x), asfloat(non_struct_raw[0].y), asfloat(non_struct_raw[0].z), asfloat(non_struct_raw[0].w));
    float _e4 = non_struct.x;
    out_.Store(0, asuint(asuint(_e4)));
    float _e10 = non_struct.w;
    out_.Store(4, asuint(asuint(_e10)));
    return;
}
