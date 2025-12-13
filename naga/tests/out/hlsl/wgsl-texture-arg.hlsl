Texture2D<float4> Texture : register(t0);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerState Sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[1]];

float4 test(Texture2D<float4> Passed_Texture, SamplerState Passed_Sampler)
{
    float4 _e5 = Passed_Texture.Sample(Passed_Sampler, float2(0.0, 0.0));
    return _e5;
}

float4 main() : SV_Target0
{
    const float4 _e2 = test(Texture, Sampler);
    return _e2;
}
