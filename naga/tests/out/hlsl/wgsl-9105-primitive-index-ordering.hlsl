struct FragmentInput_func {
    float input_location_1 : LOC0;
    uint index_1 : SV_PrimitiveID;
    float4 arbitrary_position_1 : SV_Position;
};

float4 func(FragmentInput_func fragmentinput_func) : SV_Target0
{
    float input_location = fragmentinput_func.input_location_1;
    float4 arbitrary_position = fragmentinput_func.arbitrary_position_1;
    uint index = fragmentinput_func.index_1;
    return float4(arbitrary_position.xy, input_location, float(index));
}
