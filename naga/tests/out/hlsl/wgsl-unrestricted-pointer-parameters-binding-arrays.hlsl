Texture2D<float4> textures[4] : register(t0);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const uint samplers = 0;

struct FragmentInput_main {
    float2 uv_5 : LOC0;
};

uint ZeroValueuint() {
    return (uint)0;
}

float2 ZeroValuefloat2() {
    return (float2)0;
}

float4 ZeroValuefloat4() {
    return (float4)0;
}

uint2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    float2 uv = fragmentinput_main.uv_5;
    uint i = (uint)0;
    float2 uv_1 = (float2)0;
    float4 local = (float4)0;
    float2 uv_2 = (float2)0;
    uint i_1 = (uint)0;
    float2 uv_3 = (float2)0;
    float4 local_1 = (float4)0;
    uint i_2 = (uint)0;
    float2 uv_4 = (float2)0;
    float4 local_2 = (float4)0;
    float4 local_3 = (float4)0;
    uint i_3 = (uint)0;
    uint2 local_4 = (uint2)0;

    i = 0u;
    uint _e1 = i;
    uv_1 = uv;
    float2 _e3 = uv_1;
    float4 _e8 = textures[NonUniformResourceIndex(_e1)].Sample(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samplers + _e1])], _e3);
    local = _e8;
    float4 _e10 = local;
    uv_2 = uv;
    float2 _e12 = uv_2;
    i_1 = ZeroValueuint();
    uv_3 = ZeroValuefloat2();
    local_1 = ZeroValuefloat4();
    i_2 = ZeroValueuint();
    uv_4 = ZeroValuefloat2();
    local_2 = ZeroValuefloat4();
    i_1 = 1u;
    uint _e14 = i_1;
    uv_3 = _e12;
    float2 _e16 = uv_3;
    float4 _e21 = textures[NonUniformResourceIndex(_e14)].Sample(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samplers + _e14])], _e16);
    local_1 = _e21;
    float4 _e23 = local_1;
    i_2 = 2u;
    uint _e25 = i_2;
    uv_4 = _e12;
    float2 _e27 = uv_4;
    float4 _e30 = textures[NonUniformResourceIndex(_e25)].Sample(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samplers + _e25])], _e27);
    local_2 = _e30;
    float4 _e32 = local_2;
    local_3 = (_e23 + _e32);
    float4 _e35 = local_3;
    float4 v = (_e10 + _e35);
    i_3 = 3u;
    uint _e38 = i_3;
    local_4 = NagaDimensions2D(textures[NonUniformResourceIndex(_e38)]);
    uint2 _e43 = local_4;
    return (v * float(_e43.x));
}
