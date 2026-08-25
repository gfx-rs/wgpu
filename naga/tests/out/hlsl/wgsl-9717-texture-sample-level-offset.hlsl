Texture1D<float4> sampled_texture_1d : register(t0);
Texture2D<float4> sampled_texture_2d : register(t1);
Texture3D<float4> sampled_texture_3d : register(t2);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerState texture_sampler = nagaSamplerHeap[nagaGroup0SamplerIndexArray[3]];

float4 main() : SV_Target0
{
    float4 sample_1d = sampled_texture_1d.SampleLevel(texture_sampler, 0.5, 0.0, int(int(-1)));
    float4 sample_2d = sampled_texture_2d.SampleLevel(texture_sampler, (0.5).xx, 0.0, int2(int2(int(-1), int(2))));
    float4 sample_3d = sampled_texture_3d.SampleLevel(texture_sampler, (0.5).xxx, 0.0, int3(int3(int(-1), int(2), int(-3))));
    return ((sample_1d + sample_2d) + sample_3d);
}
