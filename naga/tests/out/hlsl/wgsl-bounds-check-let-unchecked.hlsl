RWByteAddressBuffer globals : register(u0);

typedef float ret_Constructarray2_float_[2];
ret_Constructarray2_float_ Constructarray2_float_(float arg0, float arg1) {
    float ret[2] = { arg0, arg1 };
    return ret;
}

float index_array_value()
{
    float a_1[2] = Constructarray2_float_(1.0, 2.0);
    return a_1[1];
}

float index_vector_value()
{
    float2 v = float2(1.0, 2.0);
    return v[1];
}

float2 index_matrix_value()
{
    float2x2 m = float2x2(float2(1.0, 2.0), float2(3.0, 4.0));
    return m[1];
}

float index_negative_value()
{
    float2 v_1 = float2(1.0, 2.0);
    return v_1[1];
}

float index_local()
{
    float a[2] = Constructarray2_float_(1.0, 2.0);

    a[1] = 3.0;
    float _e8 = a[1];
    return _e8;
}

float index_storage()
{
    float _e4 = asfloat(globals.Load(int(9)*4+0));
    float _e8 = asfloat(globals.Load(int(9)*4+16));
    float _e14 = asfloat(globals.Load(int(9)*4+int(9)*16+32));
    return ((_e4 + _e8) + _e14);
}

void set_storage(float value)
{
    globals.Store(int(9)*4+0, asuint(value));
    globals.Store(int(9)*4+16, asuint(value));
    globals.Store(int(9)*4+int(9)*16+32, asuint(value));
    return;
}

float index_runtime_sized_negative()
{
    globals.Store(int(-1)*4+80, asuint(1.0));
    float _e8 = asfloat(globals.Load(int(-1)*4+80));
    return _e8;
}

[numthreads(1, 1, 1)]
void main()
{
    const float _e0 = index_array_value();
    const float _e1 = index_vector_value();
    const float2 _e3 = index_matrix_value();
    const float _e6 = index_negative_value();
    const float _e8 = index_local();
    const float _e10 = index_storage();
    const float _e12 = index_runtime_sized_negative();
    float sum = ((((((_e0 + _e1) + _e3.x) + _e6) + _e8) + _e10) + _e12);
    set_storage(sum);
    return;
}
