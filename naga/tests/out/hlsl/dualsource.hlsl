struct FragmentOutput {
    float4 color : SV_Target0;
    float4 mask : SV_Target1;
};

struct FragmentInput_main {
    float4 position_1 : SV_Position;
};

FragmentOutput ConstructFragmentOutput(float4 arg0, float4 arg1) {
    FragmentOutput ret = (FragmentOutput)0;
    ret.color = arg0;
    ret.mask = arg1;
    return ret;
}

FragmentOutput main(FragmentInput_main fragmentinput_main)
{
    float4 position = fragmentinput_main.position_1;
    float4 color = float4(0.4, 0.3, 0.2, 0.1);
    float4 mask = float4(0.9, 0.8, 0.7, 0.6);

    float4 _expr13 = color;
    float4 _expr14 = mask;
    const FragmentOutput fragmentoutput = ConstructFragmentOutput(_expr13, _expr14);
    return fragmentoutput;
}
