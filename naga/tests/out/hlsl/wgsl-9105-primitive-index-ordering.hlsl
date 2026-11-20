struct FragmentInput_func {
    float input_location_1 : LOC0;
    uint index_1 : SV_PrimitiveID;
    float4 arbitrary_position_1 : SV_Position;
};

float4 func(FragmentInput_func fragmentinput_func) : SV_Target0
{
    float input_location = fragmentinput_func.input_location_1;
    float4 arbitrary_position = float4(fragmentinput_func.arbitrary_position_1.xyz, 1.0 / fragmentinput_func.arbitrary_position_1.w);
    uint index = fragmentinput_func.index_1;
    float2 _e3 = arbitrary_position.xy;
    return float4(_e3.x, _e3.y, input_location, float(index));
}
