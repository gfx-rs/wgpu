struct Output {
    float4 color : SV_Target0;
    float depth : SV_DepthLessEqual;
};

struct DisagreeingOutput {
    float4 color : SV_Target0;
    float depth : SV_Depth;
};

struct FragmentInput_agree_first {
    noperspective centroid float4 pos_4 : SV_Position;
};

struct FragmentInput_agree_second {
    noperspective centroid float4 pos_5 : SV_Position;
};

struct FragmentInput_disagree_conservative {
    noperspective centroid float4 pos_6 : SV_Position;
};

struct FragmentInput_disagree_unrestricted {
    float4 pos_7 : SV_Position;
};

Output ConstructOutput(float4 arg0, float arg1) {
    Output ret = (Output)0;
    ret.color = arg0;
    ret.depth = arg1;
    return ret;
}

Output agree_first(FragmentInput_agree_first fragmentinput_agree_first)
{
    float4 pos = fragmentinput_agree_first.pos_4;
    const Output output = ConstructOutput(float4(0.4, 0.3, 0.2, 0.1), (pos.z - 0.1));
    return output;
}

Output agree_second(FragmentInput_agree_second fragmentinput_agree_second)
{
    float4 pos_1 = fragmentinput_agree_second.pos_5;
    const Output output_1 = ConstructOutput(float4(0.1, 0.2, 0.3, 0.4), (pos_1.z - 0.2));
    return output_1;
}

DisagreeingOutput ConstructDisagreeingOutput(float4 arg0, float arg1) {
    DisagreeingOutput ret = (DisagreeingOutput)0;
    ret.color = arg0;
    ret.depth = arg1;
    return ret;
}

DisagreeingOutput disagree_conservative(FragmentInput_disagree_conservative fragmentinput_disagree_conservative)
{
    float4 pos_2 = fragmentinput_disagree_conservative.pos_6;
    const DisagreeingOutput disagreeingoutput = ConstructDisagreeingOutput(float4(0.4, 0.3, 0.2, 0.1), (pos_2.z + 0.1));
    return disagreeingoutput;
}

DisagreeingOutput disagree_unrestricted(FragmentInput_disagree_unrestricted fragmentinput_disagree_unrestricted)
{
    float4 pos_3 = fragmentinput_disagree_unrestricted.pos_7;
    const DisagreeingOutput disagreeingoutput_1 = ConstructDisagreeingOutput(float4(0.1, 0.2, 0.3, 0.4), (pos_3.z * 2.0));
    return disagreeingoutput_1;
}
