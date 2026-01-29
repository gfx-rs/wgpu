// Used for HLSL, DXIL, and SPIRV passthrough

struct VSOut
{
    float4 position : SV_POSITION;
};

#ifdef SPIRV
[shader("vertex")]
#endif
VSOut vertex_main(uint vid: SV_VertexID)
{
    VSOut output;

    float2 positions[3] = {
        float2(0.0, 0.5),
        float2(-0.5, -0.5),
        float2(0.5, -0.5),
    };

    output.position = float4(positions[vid], 0.0, 1.0);
    return output;
}

#ifdef SPIRV
[shader("pixel")]
#endif
float4 fragment_main(VSOut input) : SV_TARGET
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
