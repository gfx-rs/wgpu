static const float c_scale = 1.2;

Texture2D<float4> u_texture : register(t0);
SamplerState u_sampler : register(s1);

struct VertexOutput {
    float2 uv : LOC0;
    float4 position : SV_Position;
};

struct VertexInput {
    float2 pos1 : LOC0;
    float2 uv3 : LOC1;
};

struct FragmentInput {
    float2 uv4 : LOC0;
};

VertexOutput vert_main(VertexInput vertexinput)
{
    const VertexOutput vertexoutput1 = { vertexinput.uv3, float4((1.2 * vertexinput.pos1), 0.0, 1.0) };
    return vertexoutput1;
}

float4 frag_main(FragmentInput fragmentinput) : SV_Target0
{
    float4 color = u_texture.Sample(u_sampler, fragmentinput.uv4);
    if ((color.w == 0.0)) {
        discard;
    }
    float4 premultiplied = (color.w * color);
    return premultiplied;
}
