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

    float4 _expr1 = ddx_coarse(foo);
    x = _expr1;
    float4 _expr3 = ddy_coarse(foo);
    y = _expr3;
    float4 _expr5 = abs(ddx_coarse(foo)) + abs(ddy_coarse(foo));
    z = _expr5;
    float4 _expr7 = ddx_fine(foo);
    x = _expr7;
    float4 _expr8 = ddy_fine(foo);
    y = _expr8;
    float4 _expr9 = abs(ddx_fine(foo)) + abs(ddy_fine(foo));
    z = _expr9;
    float4 _expr10 = ddx(foo);
    x = _expr10;
    float4 _expr11 = ddy(foo);
    y = _expr11;
    float4 _expr12 = fwidth(foo);
    z = _expr12;
    const bool _e13 = test_any_and_all_for_bool();
    float4 _expr14 = x;
    float4 _expr15 = y;
    float4 _expr17 = z;
    return ((_expr14 + _expr15) * _expr17);
}
