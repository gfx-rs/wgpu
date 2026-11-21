struct PackedVec3_ {
    float a;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    float3 c;
    float d;
};

struct Matrix {
    int a;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    int4 c;
    float2 d_0; float2 d_1; float2 d_2;
    int _end_pad_0;
    int _end_pad_1;
};

struct TrailingPad {
    float2 m_0; float2 m_1;
    float a;
    int _end_pad_0;
};

PackedVec3_ ConstructPackedVec3_(float arg0, float3 arg1, float arg2) {
    PackedVec3_ ret = (PackedVec3_)0;
    ret.a = arg0;
    ret.c = arg1;
    ret.d = arg2;
    return ret;
}

Matrix ConstructMatrix(int arg0, int4 arg1, float3x2 arg2) {
    Matrix ret = (Matrix)0;
    ret.a = arg0;
    ret.c = arg1;
    ret.d_0 = arg2[0];
    ret.d_1 = arg2[1];
    ret.d_2 = arg2[2];
    return ret;
}

TrailingPad ConstructTrailingPad(float2x2 arg0, float arg1) {
    TrailingPad ret = (TrailingPad)0;
    ret.m_0 = arg0[0];
    ret.m_1 = arg0[1];
    ret.a = arg1;
    return ret;
}

static const PackedVec3_ PACKED_VEC3_ = ConstructPackedVec3_(1.0, (2.0).xxx, 3.0);
static const Matrix MATRIX = ConstructMatrix(int(1), (int(3)).xxxx, float3x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0)));
static const TrailingPad TRAILING_PAD = ConstructTrailingPad(float2x2(float2(1.0, 2.0), float2(3.0, 4.0)), 5.0);

RWByteAddressBuffer out_ : register(u0);

float packed_vec3_(PackedVec3_ s)
{
    return ((s.a + s.c.x) + s.d);
}

float3x2 GetMatdOnMatrix(Matrix obj) {
    return float3x2(obj.d_0, obj.d_1, obj.d_2);
}

void SetMatdOnMatrix(Matrix obj, float3x2 mat) {
    obj.d_0 = mat[0];
    obj.d_1 = mat[1];
    obj.d_2 = mat[2];
}

void SetMatVecdOnMatrix(Matrix obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.d_0 = vec; break; }
    case 1: { obj.d_1 = vec; break; }
    case 2: { obj.d_2 = vec; break; }
    }
}

void SetMatScalardOnMatrix(Matrix obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.d_0[vec_idx] = scalar; break; }
    case 1: { obj.d_1[vec_idx] = scalar; break; }
    case 2: { obj.d_2[vec_idx] = scalar; break; }
    }
}

float matrix_(Matrix s_1)
{
    return ((float(s_1.a) + float(s_1.c.x)) + GetMatdOnMatrix(s_1)[0].x);
}

float2x2 GetMatmOnTrailingPad(TrailingPad obj) {
    return float2x2(obj.m_0, obj.m_1);
}

void SetMatmOnTrailingPad(TrailingPad obj, float2x2 mat) {
    obj.m_0 = mat[0];
    obj.m_1 = mat[1];
}

void SetMatVecmOnTrailingPad(TrailingPad obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0 = vec; break; }
    case 1: { obj.m_1 = vec; break; }
    }
}

void SetMatScalarmOnTrailingPad(TrailingPad obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0[vec_idx] = scalar; break; }
    case 1: { obj.m_1[vec_idx] = scalar; break; }
    }
}

float trailing_pad(TrailingPad s_2)
{
    return (GetMatmOnTrailingPad(s_2)[0].x + s_2.a);
}

[numthreads(1, 1, 1)]
void main()
{
    const float _e3 = packed_vec3_(PACKED_VEC3_);
    out_.Store(0, asuint(_e3));
    const float _e7 = matrix_(MATRIX);
    out_.Store(4, asuint(_e7));
    const float _e11 = trailing_pad(TRAILING_PAD);
    out_.Store(8, asuint(_e11));
    return;
}
