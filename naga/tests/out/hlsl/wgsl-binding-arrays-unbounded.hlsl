Texture2D<float4> textures[] : register(t0);

struct FragmentInput_main {
    nointerpolation uint index_1 : LOC0;
};

uint2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    uint index = fragmentinput_main.index_1;
    uint2 dim = NagaDimensions2D(textures[NonUniformResourceIndex(index)]);
    return float4(float(dim.x), float(dim.y), 0.0, 1.0);
}
