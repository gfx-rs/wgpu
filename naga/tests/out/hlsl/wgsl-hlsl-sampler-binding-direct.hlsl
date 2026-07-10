struct FragmentIn {
    float2 uv : LOC0;
    nointerpolation uint index : LOC1;
};

Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
Texture2D<float> depth_tex : register(t1);
SamplerComparisonState samp_comp : register(s1);
SamplerState samp_array[4] : register(s2);

struct FragmentInput_main {
    float2 uv : LOC0;
    nointerpolation uint index : LOC1;
};

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn in_ = { fragmentinput_main.uv, fragmentinput_main.index };
    float4 color = (0.0).xxxx;

    float4 _e4 = color;
    float4 _e8 = tex.Sample(samp, in_.uv);
    color = (_e4 + _e8);
    float _e11 = color.x;
    float _e16 = depth_tex.SampleCmp(samp_comp, in_.uv, 0.5);
    color.x = (_e11 + _e16);
    float4 _e18 = color;
    float4 _e23 = tex.Sample(samp_array[0], in_.uv);
    color = (_e18 + _e23);
    float4 _e25 = color;
    float4 _e31 = tex.Sample(samp_array[NonUniformResourceIndex(in_.index)], in_.uv);
    color = (_e25 + _e31);
    float4 _e33 = color;
    return _e33;
}
