RWTexture3D<unorm float4> inputTexture : register(u0);
RWTexture3D<unorm float4> outputTexture : register(u1);

uint3 NagaRWDimensions3D(RWTexture3D<unorm float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.xyz;
}

void main()
{
    uint3 srcSize = NagaRWDimensions3D(inputTexture);
    uint3 dstSize = NagaRWDimensions3D(outputTexture);
    return;
}
