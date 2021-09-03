static const float c_scale = 1.2;

struct VertexOutput {
    linear float2 uv : LOC0;
    float4 position : SV_Position;
};

Texture2D<float4> u_texture : register(t0);
SamplerState u_sampler : register(s1);

struct VertexOutput_main {
    float2 uv2 : LOC0;
    float4 position : SV_Position;
};

struct FragmentInput_main {
    float2 uv3 : LOC0;
};

VertexOutput ConstructVertexOutput(float2 arg0, float4 arg1) {
    VertexOutput ret;
    ret.uv = arg0;
    ret.position = arg1;
    return ret;
}

VertexOutput_main main(float2 pos : LOC0, float2 uv : LOC1)
{
    const VertexOutput vertexoutput = ConstructVertexOutput(uv, float4((c_scale * pos), 0.0, 1.0));
    const VertexOutput_main vertexoutput1 = { vertexoutput.uv, vertexoutput.position };
    return vertexoutput1;
}

float4 main1(FragmentInput_main fragmentinput_main) : SV_Target0
{
    float2 uv1 = fragmentinput_main.uv3;
    float4 color = u_texture.Sample(u_sampler, uv1);
    if ((color.w == 0.0)) {
        discard;
    }
    float4 premultiplied = (color.w * color);
    return premultiplied;
}

float4 fs_extra() : SV_Target0
{
    return float4(0.0, 0.5, 0.0, 0.5);
}
