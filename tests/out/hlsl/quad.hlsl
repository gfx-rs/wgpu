static const float c_scale = 1.2;

struct VertexOutput {
    float2 uv : LOC0;
    float4 position : SV_Position;
};

Texture2D<float4> u_texture : register(t0);
SamplerState u_sampler : register(s1);

struct VertexInput {
    float2 pos1 : LOC0;
    float2 uv2 : LOC1;
};

struct FragmentInput {
    float2 uv3 : LOC0;
};

VertexOutput vert_main(VertexInput vertexinput)
{
    const VertexOutput vertexoutput1 = { vertexinput.uv2, float4((c_scale * vertexinput.pos1), 0.0, 1.0) };
    return vertexoutput1;
}

float4 frag_main(FragmentInput fragmentinput) : SV_Target0
{
    float4 color = u_texture.Sample(u_sampler, fragmentinput.uv3);
    if ((color.w == 0.0)) {
        discard;
    }
    float4 premultiplied = (color.w * color);
    return premultiplied;
}
