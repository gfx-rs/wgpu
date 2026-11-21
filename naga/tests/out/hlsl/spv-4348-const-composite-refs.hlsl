struct Packed {
    float3 member;
    float member_1;
};

typedef float2x2 ret_Constructarray2_float2x2_[2];
ret_Constructarray2_float2x2_ Constructarray2_float2x2_(float2x2 arg0, float2x2 arg1) {
    float2x2 ret[2] = { arg0, arg1 };
    return ret;
}

Packed ConstructPacked(float3 arg0, float arg1) {
    Packed ret = (Packed)0;
    ret.member = arg0;
    ret.member_1 = arg1;
    return ret;
}

typedef Packed ret_Constructarray2_Packed_[2];
ret_Constructarray2_Packed_ Constructarray2_Packed_(Packed arg0, Packed arg1) {
    Packed ret[2] = { arg0, arg1 };
    return ret;
}

static const float2 VEC = float2(1.0, 2.0);
static const float2x2 MATRIX = float2x2(VEC, VEC);
static const float2x2 MATRICES[2] = Constructarray2_float2x2_(MATRIX, MATRIX);
static const Packed PACKED = ConstructPacked(float3(1.0, 1.0, 1.0), 2.0);
static const Packed PACKED_ARRAY[2] = Constructarray2_Packed_(PACKED, PACKED);

static float4 color = (float4)0;

void main_1()
{
    color = float4(MATRICES[1][0].y, PACKED_ARRAY[1].member.z, PACKED_ARRAY[0].member_1, 1.0);
    return;
}

float4 main() : SV_Target0
{
    main_1();
    float4 _e1 = color;
    return _e1;
}
