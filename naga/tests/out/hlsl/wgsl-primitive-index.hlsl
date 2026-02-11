struct FragmentInput_func {
    uint index_1 : SV_PrimitiveID;
};

float4 func(FragmentInput_func fragmentinput_func) : SV_Target0
{
    uint index = fragmentinput_func.index_1;
    return float4(float(index), 1.0, 1.0, 1.0);
}
