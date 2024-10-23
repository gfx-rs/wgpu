// slangc bin

[[vk::binding(0, 0)]]
Texture2D textures[];

[[vk::binding(0, 1)]]
SamplerState samplers[];

[[vk::binding(0, 2)]]
Texture2D cmp_texture;

[[vk::binding(0, 3)]]
SamplerComparisonState cmp_samplers[];

[[vk::binding(0, 4)]]
RWTexture2D rw_textures[];

[shader("fragment")]
float4 main(uint tex_id, uint samp_id, float2 uv, int3 location, float lod, float clamp, float cmp, float2 ddx, float2 ddy)
    : SV_Target
{
    float4 out = float4(0.0);
    out += textures[tex_id].Sample(samplers[samp_id], uv);
    out += textures[tex_id].Sample(samplers[samp_id], uv, int2(1.0));
    out += textures[tex_id].Sample(samplers[samp_id], uv, int2(1.0), clamp);
    out += textures[tex_id].SampleLevel(samplers[samp_id], uv, lod);
    out += textures[tex_id].SampleLevel(samplers[samp_id], uv, lod, int2(1.0));
    out += cmp_texture.SampleCmp(cmp_samplers[samp_id], uv, cmp);
    out += cmp_texture.SampleCmp(cmp_samplers[samp_id], uv, cmp, int2(1.0));
    out += cmp_texture.SampleCmpLevelZero(cmp_samplers[samp_id], uv, cmp);
    out += cmp_texture.SampleCmpLevelZero(cmp_samplers[samp_id], uv, cmp, int2(1.0));
    out += textures[tex_id].SampleGrad(samplers[samp_id], uv, ddx, ddy);
    out += textures[tex_id].SampleGrad(samplers[samp_id], uv, ddx, ddy, int2(1.0));
    out += textures[tex_id].SampleGrad(samplers[samp_id], uv, ddx, ddy, int2(1.0), clamp);

    out += textures[tex_id][location.xy];
    out += textures[tex_id].Load(location, int2(1.0));
    out += rw_textures[tex_id].Load(location.xy);
    out += rw_textures[tex_id].Load(location.xy, int2(1.0));

    rw_textures[tex_id][location.xy] = float4(1.0);

    return out;
};
