struct FragmentInput_derivatives {
    float4 foo_1 : SV_Position;
};

float4 derivatives(FragmentInput_derivatives fragmentinput_derivatives) : SV_Target0
{
    float4 foo = float4(fragmentinput_derivatives.foo_1.xyz, 1.0 / fragmentinput_derivatives.foo_1.w);
    float4 x = (float4)0;
    float4 y = (float4)0;
    float4 z = (float4)0;
    bool local = (bool)0;

    float4 _e1 = ddx_coarse(foo);
    x = _e1;
    float4 _e2 = ddy_coarse(foo);
    y = _e2;
    float4 _e3 = abs(ddx_coarse(foo)) + abs(ddy_coarse(foo));
    z = _e3;
    float4 _e4 = ddx_fine(foo);
    x = _e4;
    float4 _e5 = ddy_fine(foo);
    y = _e5;
    float4 _e6 = abs(ddx_fine(foo)) + abs(ddy_fine(foo));
    z = _e6;
    float4 _e7 = ddx(foo);
    x = _e7;
    float4 _e8 = ddy(foo);
    y = _e8;
    float4 _e9 = fwidth(foo);
    z = _e9;
    local = true;
    bool a = local;
    float4 _e15 = x;
    float4 _e16 = y;
    float4 _e18 = z;
    return ((_e15 + _e16) * _e18);
}
