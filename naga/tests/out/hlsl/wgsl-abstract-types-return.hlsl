int return_i32_ai()
{
    return int(1);
}

uint return_u32_ai()
{
    return 1u;
}

float return_f32_ai()
{
    return 1.0;
}

float return_f32_af()
{
    return 1.0;
}

float2 return_vec2f32_ai()
{
    return (1.0).xx;
}

typedef float ret_Constructarray4_float_[4];
ret_Constructarray4_float_ Constructarray4_float_(float arg0, float arg1, float arg2, float arg3) {
    float ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef float ret_return_arrf32_ai[4];
ret_return_arrf32_ai return_arrf32_ai()
{
    return Constructarray4_float_(1.0, 1.0, 1.0, 1.0);
}

float return_const_f32_const_ai()
{
    return 1.0;
}

float2 return_vec2f32_const_ai()
{
    return (1.0).xx;
}

[numthreads(1, 1, 1)]
void main()
{
    return;
}
