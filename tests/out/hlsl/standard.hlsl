
struct FragmentInput_derivatives {
    float4 foo_1 : SV_Position;
};

float4 derivatives(FragmentInput_derivatives fragmentinput_derivatives) : SV_Target0
{
    float4 foo = fragmentinput_derivatives.foo_1;
    float4 x = ddx(foo);
    float4 y = ddy(foo);
    float4 z = fwidth(foo);
    return ((x + y) * z);
}
