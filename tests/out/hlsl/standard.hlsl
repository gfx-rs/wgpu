
struct FragmentInput_derivatives {
    float4 foo1 : SV_Position;
};

float4 derivatives(FragmentInput_derivatives fragmentinput_derivatives) : SV_Target0
{
    float4 x = ddx(fragmentinput_derivatives.foo1);
    float4 y = ddy(fragmentinput_derivatives.foo1);
    float4 z = fwidth(fragmentinput_derivatives.foo1);
    return ((x + y) * z);
}
