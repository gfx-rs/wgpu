typedef float ret_Constructarray2_float_[2];
ret_Constructarray2_float_ Constructarray2_float_(float arg0, float arg1) {
    float ret[2] = { arg0, arg1 };
    return ret;
}

typedef float ret_ret_array[2];
ret_ret_array ret_array()
{
    return Constructarray2_float_(1.0, 2.0);
}

typedef float ret_Constructarray3_array2_float__[3][2];
ret_Constructarray3_array2_float__ Constructarray3_array2_float__(float arg0[2], float arg1[2], float arg2[2]) {
    float ret[3][2] = { arg0, arg1, arg2 };
    return ret;
}

typedef float ret_ret_array_array[3][2];
ret_ret_array_array ret_array_array()
{
    const float _e0[2] = ret_array();
    const float _e1[2] = ret_array();
    const float _e2[2] = ret_array();
    return Constructarray3_array2_float__(_e0, _e1, _e2);
}

float4 main() : SV_Target0
{
    const float _e0[3][2] = ret_array_array();
    return float4(_e0[0][0], _e0[0][1], 0.0, 1.0);
}
