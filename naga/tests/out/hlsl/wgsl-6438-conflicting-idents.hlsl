struct OurVertexShaderOutput {
    float4 position : SV_Position;
    float2 texcoord : LOC0;
};

struct VertexOutput_vs {
    float2 texcoord : LOC0;
    float4 position : SV_Position;
};

VertexOutput_vs vs(float2 xy : LOC0)
{
    OurVertexShaderOutput vsOutput = (OurVertexShaderOutput)0;

    vsOutput.position = float4(xy.x, xy.y, 0.0, 1.0);
    OurVertexShaderOutput _e8 = vsOutput;
    const OurVertexShaderOutput ourvertexshaderoutput = _e8;
    const VertexOutput_vs ourvertexshaderoutput_1 = { ourvertexshaderoutput.texcoord, ourvertexshaderoutput.position };
    return ourvertexshaderoutput_1;
}

float4 fs() : SV_Target0
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
