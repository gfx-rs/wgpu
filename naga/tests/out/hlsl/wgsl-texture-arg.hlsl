Texture2D<float4> Texture : register(t0);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerState Sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[1]];

float4 main() : SV_Target0
{
    float4 local = (float4)0;

    float4 _e5 = Texture.Sample(Sampler, float2(0.0, 0.0));
    local = _e5;
    float4 _e7 = local;
    return _e7;
}
