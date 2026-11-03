struct ColorAndDepth {
    float4 color : SV_Target0;
    float depth : SV_Depth;
};

struct OnlyDepth {
    float depth : SV_Depth;
};

struct FragmentInput_color_and_depth {
    float4 pos_3 : SV_Position;
};

struct FragmentOutput_color_and_depth {
    float4 color : SV_Target0;
};

struct FragmentInput_only_depth {
    float4 pos_4 : SV_Position;
};

struct FragmentInput_unrestricted {
    float4 pos_5 : SV_Position;
};

ColorAndDepth ConstructColorAndDepth(float4 arg0, float arg1) {
    ColorAndDepth ret = (ColorAndDepth)0;
    ret.color = arg0;
    ret.depth = arg1;
    return ret;
}

FragmentOutput_color_and_depth color_and_depth(FragmentInput_color_and_depth fragmentinput_color_and_depth)
{
    float4 pos = fragmentinput_color_and_depth.pos_3;
    const ColorAndDepth coloranddepth = ConstructColorAndDepth(float4(0.4, 0.3, 0.2, 0.1), pos.z);
    const FragmentOutput_color_and_depth coloranddepth_1 = { coloranddepth.color };
    return coloranddepth_1;
}

OnlyDepth ConstructOnlyDepth(float arg0) {
    OnlyDepth ret = (OnlyDepth)0;
    ret.depth = arg0;
    return ret;
}

void only_depth(FragmentInput_only_depth fragmentinput_only_depth)
{
    float4 pos_1 = fragmentinput_only_depth.pos_4;
    return;
}

ColorAndDepth unrestricted(FragmentInput_unrestricted fragmentinput_unrestricted)
{
    float4 pos_2 = fragmentinput_unrestricted.pos_5;
    const ColorAndDepth coloranddepth_2 = ConstructColorAndDepth(float4(0.1, 0.2, 0.3, 0.4), (pos_2.z * 2.0));
    return coloranddepth_2;
}
