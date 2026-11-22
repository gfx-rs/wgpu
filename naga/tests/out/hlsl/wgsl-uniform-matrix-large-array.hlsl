struct mat2x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
};

cbuffer a : register(b0) { mat2x2f32_std140_ a[64]; }
cbuffer b : register(b1) { mat2x2f32_std140_ b[20][2]; }
RWByteAddressBuffer out_ : register(u2);

typedef float2x2 ret_Constructarray2_float2x2_[2];
ret_Constructarray2_float2x2_ Constructarray2_float2x2_(float2x2 arg0, float2x2 arg1) {
    float2x2 ret[2] = { arg0, arg1 };
    return ret;
}

typedef mat2x2f32_std140_ ret_ZeroValuearray64_mat2x2f32_std140__[64];
ret_ZeroValuearray64_mat2x2f32_std140__ ZeroValuearray64_mat2x2f32_std140__() {
    return (mat2x2f32_std140_[64])0;
}

typedef float2x2 ret_ZeroValuearray64_float2x2_[64];
ret_ZeroValuearray64_float2x2_ ZeroValuearray64_float2x2_() {
    return (float2x2[64])0;
}

typedef mat2x2f32_std140_ ret_ZeroValuearray20_array2_mat2x2f32_std140___[20][2];
ret_ZeroValuearray20_array2_mat2x2f32_std140___ ZeroValuearray20_array2_mat2x2f32_std140___() {
    return (mat2x2f32_std140_[20][2])0;
}

typedef float2x2 ret_ZeroValuearray20_array2_float2x2__[20][2];
ret_ZeroValuearray20_array2_float2x2__ ZeroValuearray20_array2_float2x2__() {
    return (float2x2[20][2])0;
}

typedef mat2x2f32_std140_ ret_ZeroValuearray2_mat2x2f32_std140__[2];
ret_ZeroValuearray2_mat2x2f32_std140__ ZeroValuearray2_mat2x2f32_std140__() {
    return (mat2x2f32_std140_[2])0;
}

typedef float2x2 ret_ZeroValuearray2_float2x2_[2];
ret_ZeroValuearray2_float2x2_ ZeroValuearray2_float2x2_() {
    return (float2x2[2])0;
}

[numthreads(1, 1, 1)]
void main(uint i : SV_GroupIndex)
{
    mat2x2f32_std140_ v[64] = (mat2x2f32_std140_[64])0;
    mat2x2f32_std140_ s[64] = (mat2x2f32_std140_[64])0;
    float2x2 r[64] = (float2x2[64])0;
    uint i_1 = (uint)0;
    float2x2 local[64] = (float2x2[64])0;
    mat2x2f32_std140_ v_1[20][2] = (mat2x2f32_std140_[20][2])0;
    mat2x2f32_std140_ s_1[20][2] = (mat2x2f32_std140_[20][2])0;
    float2x2 r_1[20][2] = (float2x2[20][2])0;
    uint i_2 = (uint)0;
    mat2x2f32_std140_ v_2[2] = (mat2x2f32_std140_[2])0;
    float2x2 local_1[2] = (float2x2[2])0;
    float2x2 local_2[20][2] = (float2x2[20][2])0;

    mat2x2f32_std140_ _e1[64] = a;
    v = _e1;
    mat2x2f32_std140_ _e3[64] = v;
    s = ZeroValuearray64_mat2x2f32_std140__();
    r = ZeroValuearray64_float2x2_();
    i_1 = 0u;
    s = _e3;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e17 = i_1;
            i_1 = (_e17 + 1u);
        }
        loop_init = false;
        uint _e6 = i_1;
        if ((_e6 < 64u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e11 = s[min(uint(_e6), 63u)];
        r[min(uint(_e6), 63u)] = float2x2(_e11.c_0_, _e11.c_1_);
    }
    float2x2 _e19[64] = r;
    local = _e19;
    float2x2 x[64] = local;
    mat2x2f32_std140_ _e23[20][2] = b;
    v_1 = _e23;
    mat2x2f32_std140_ _e25[20][2] = v_1;
    s_1 = ZeroValuearray20_array2_mat2x2f32_std140___();
    r_1 = ZeroValuearray20_array2_float2x2__();
    i_2 = 0u;
    v_2 = ZeroValuearray2_mat2x2f32_std140__();
    local_1 = ZeroValuearray2_float2x2_();
    s_1 = _e25;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e49 = i_2;
            i_2 = (_e49 + 1u);
        }
        loop_init_1 = false;
        uint _e28 = i_2;
        if ((_e28 < 20u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e32[2] = s_1[min(uint(_e28), 19u)];
        v_2 = _e32;
        mat2x2f32_std140_ _e34[2] = v_2;
        local_1 = Constructarray2_float2x2_(float2x2(_e34[0].c_0_, _e34[0].c_1_), float2x2(_e34[1].c_0_, _e34[1].c_1_));
        float2x2 _e45[2] = local_1;
        r_1[min(uint(_e28), 19u)] = _e45;
    }
    float2x2 _e51[20][2] = r_1;
    local_2 = _e51;
    float2x2 y[20][2] = local_2;
    {
        float2x2 _value2 = x[min(uint(i), 63u)];
        out_.Store2(0+0, asuint(_value2[0]));
        out_.Store2(0+8, asuint(_value2[1]));
    }
    {
        float2x2 _value2 = y[min(uint(i), 19u)][1];
        out_.Store2(16+0, asuint(_value2[0]));
        out_.Store2(16+8, asuint(_value2[1]));
    }
    return;
}
