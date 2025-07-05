struct FragmentOutput {
    float4 output0_ : SV_Target0;
    float4 output1_ : SV_Target1;
};

FragmentOutput ConstructFragmentOutput(float4 arg0, float4 arg1) {
    FragmentOutput ret = (FragmentOutput)0;
    ret.output0_ = arg0;
    ret.output1_ = arg1;
    return ret;
}

FragmentOutput main()
{
    const FragmentOutput fragmentoutput = ConstructFragmentOutput(float4(0.4, 0.3, 0.2, 0.1), float4(0.9, 0.8, 0.7, 0.6));
    return fragmentoutput;
}
