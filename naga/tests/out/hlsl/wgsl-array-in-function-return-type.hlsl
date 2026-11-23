typedef float ret_Constructarray2_float_[2];
ret_Constructarray2_float_ Constructarray2_float_(float arg0, float arg1) {
    float ret[2] = { arg0, arg1 };
    return ret;
}

typedef float ret_Constructarray3_array2_float__[3][2];
ret_Constructarray3_array2_float__ Constructarray3_array2_float__(float arg0[2], float arg1[2], float arg2[2]) {
    float ret[3][2] = { arg0, arg1, arg2 };
    return ret;
}

typedef float ret_ZeroValuearray2_float_[2];
ret_ZeroValuearray2_float_ ZeroValuearray2_float_() {
    return (float[2])0;
}

float4 main() : SV_Target0
{
    float local[2] = (float[2])0;
    float local_1[2] = (float[2])0;
    float local_2[2] = (float[2])0;
    float local_3[3][2] = (float[3][2])0;

    local = ZeroValuearray2_float_();
    local_1 = ZeroValuearray2_float_();
    local_2 = ZeroValuearray2_float_();
    local = Constructarray2_float_(1.0, 2.0);
    float _e4[2] = local;
    local_1 = Constructarray2_float_(1.0, 2.0);
    float _e9[2] = local_1;
    local_2 = Constructarray2_float_(1.0, 2.0);
    float _e14[2] = local_2;
    local_3 = Constructarray3_array2_float__(_e4, _e9, _e14);
    float a[3][2] = local_3;
    return float4(a[0][0], a[0][1], 0.0, 1.0);
}
