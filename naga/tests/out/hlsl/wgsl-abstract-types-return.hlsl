typedef float ret_Constructarray4_float_[4];
ret_Constructarray4_float_ Constructarray4_float_(float arg0, float arg1, float arg2, float arg3) {
    float ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    int local = (int)0;
    uint local_1 = (uint)0;
    float local_2 = (float)0;
    float local_3 = (float)0;
    float2 local_4 = (float2)0;
    float local_5[4] = (float[4])0;
    float local_6 = (float)0;
    float2 local_7 = (float2)0;

    local = int(1);
    local_1 = 1u;
    local_2 = 1.0;
    local_3 = 1.0;
    local_4 = (1.0).xx;
    local_5 = Constructarray4_float_(1.0, 1.0, 1.0, 1.0);
    local_6 = 1.0;
    local_7 = (1.0).xx;
    return;
}
