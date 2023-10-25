Texture2D<float4> Texture : register(t0);
SamplerState Sampler : register(s1);

float4 test(Texture2D<float4> Passed_Texture, SamplerState Passed_Sampler)
{
    float4 _expr5 = Passed_Texture.Sample(Passed_Sampler, float2(0.0, 0.0));
    return _expr5;
}

float4 main() : SV_Target0
{
    const float4 _e2 = test(Texture, Sampler);
    return _e2;
}
