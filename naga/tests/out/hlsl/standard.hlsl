struct FragmentInput_derivatives {
    float4 foo_1 : SV_Position;
};

bool test_any_and_all_for_bool()
{
    return true;
}

float4 derivatives(FragmentInput_derivatives fragmentinput_derivatives) : SV_Target0
{
    float4 foo = fragmentinput_derivatives.foo_1;
    float4 x = (float4)0;
    float4 y = (float4)0;
    float4 z = (float4)0;

    float4 _e1 = ddx_coarse(foo);
    x = _e1;
    float4 _e3 = ddy_coarse(foo);
    y = _e3;
    float4 _e5 = abs(ddx_coarse(foo)) + abs(ddy_coarse(foo));
    z = _e5;
    float4 _e7 = ddx_fine(foo);
    x = _e7;
    float4 _e8 = ddy_fine(foo);
    y = _e8;
    float4 _e9 = abs(ddx_fine(foo)) + abs(ddy_fine(foo));
    z = _e9;
    float4 _e10 = ddx(foo);
    x = _e10;
    float4 _e11 = ddy(foo);
    y = _e11;
    float4 _e12 = fwidth(foo);
    z = _e12;
    const bool _e13 = test_any_and_all_for_bool();
    float4 _e14 = x;
    float4 _e15 = y;
    float4 _e17 = z;
    return ((_e14 + _e15) * _e17);
}
