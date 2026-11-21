struct mat2x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
};

cbuffer a : register(b0) { mat2x2f32_std140_ a[64]; }
cbuffer b : register(b1) { mat2x2f32_std140_ b[20][2]; }
RWByteAddressBuffer out_ : register(u2);

typedef float2x2 ret_convert_std140_0_[64];
ret_convert_std140_0_ convert_std140_0_(mat2x2f32_std140_ v[64])
{
    mat2x2f32_std140_ s[64] = (mat2x2f32_std140_[64])0;
    float2x2 r[64] = (float2x2[64])0;
    uint i_1 = 0u;

    s = v;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e15 = i_1;
            i_1 = (_e15 + 1u);
        }
        loop_init = false;
        uint _e7 = i_1;
        if ((_e7 < 64u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e10 = s[min(uint(_e7), 63u)];
        r[min(uint(_e7), 63u)] = float2x2(_e10.c_0_, _e10.c_1_);
    }
    float2x2 _e17[64] = r;
    return _e17;
}

typedef float2x2 ret_Constructarray2_float2x2_[2];
ret_Constructarray2_float2x2_ Constructarray2_float2x2_(float2x2 arg0, float2x2 arg1) {
    float2x2 ret[2] = { arg0, arg1 };
    return ret;
}

typedef float2x2 ret_convert_std140_1_[2];
ret_convert_std140_1_ convert_std140_1_(mat2x2f32_std140_ v_1[2])
{
    return Constructarray2_float2x2_(float2x2(v_1[0].c_0_, v_1[0].c_1_), float2x2(v_1[1].c_0_, v_1[1].c_1_));
}

typedef float2x2 ret_convert_std140_2_[20][2];
ret_convert_std140_2_ convert_std140_2_(mat2x2f32_std140_ v_2[20][2])
{
    mat2x2f32_std140_ s_1[20][2] = (mat2x2f32_std140_[20][2])0;
    float2x2 r_1[20][2] = (float2x2[20][2])0;
    uint i_2 = 0u;

    s_1 = v_2;
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e13 = i_2;
            i_2 = (_e13 + 1u);
        }
        loop_init_1 = false;
        uint _e7 = i_2;
        if ((_e7 < 20u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e10[2] = s_1[min(uint(_e7), 19u)];
        const float2x2 _e11[2] = convert_std140_1_(_e10);
        r_1[min(uint(_e7), 19u)] = _e11;
    }
    float2x2 _e15[20][2] = r_1;
    return _e15;
}

[numthreads(1, 1, 1)]
void main(uint i : SV_GroupIndex)
{
    mat2x2f32_std140_ _e12[64] = a;
    const float2x2 _e2[64] = convert_std140_0_(_e12);
    mat2x2f32_std140_ _e13[20][2] = b;
    const float2x2 _e4[20][2] = convert_std140_2_(_e13);
    {
        float2x2 _value2 = _e2[min(uint(i), 63u)];
        out_.Store2(0+0, asuint(_value2[0]));
        out_.Store2(0+8, asuint(_value2[1]));
    }
    {
        float2x2 _value2 = _e4[min(uint(i), 19u)][1];
        out_.Store2(16+0, asuint(_value2[0]));
        out_.Store2(16+8, asuint(_value2[1]));
    }
    return;
}
