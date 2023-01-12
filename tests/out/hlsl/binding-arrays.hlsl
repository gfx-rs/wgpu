
struct UniformIndex {
    uint index;
};

struct FragmentIn {
    nointerpolation uint index : LOC0;
};

Texture2D<float4> texture_array_unbounded[10] : register(t0);
Texture2D<float4> texture_array_bounded[5] : register(t0, space1);
Texture2DArray<float4> texture_array_2darray[5] : register(t0, space2);
Texture2DMS<float4> texture_array_multisampled[5] : register(t0, space3);
Texture2D<float> texture_array_depth[5] : register(t0, space4);
RWTexture2D<float4> texture_array_storage[5] : register(u0, space5);
SamplerState samp[5] : register(s0, space6);
SamplerComparisonState samp_comp[5] : register(s0, space7);
cbuffer uni : register(b0, space8) { UniformIndex uni; }

struct FragmentInput_main {
    nointerpolation uint index : LOC0;
};

int2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

int NagaNumLayers2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaNumLevels2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

int NagaMSNumSamples2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.z;
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn fragment_in = { fragmentinput_main.index };
    int i1_ = (int)0;
    int2 i2_ = (int2)0;
    float v1_ = (float)0;
    float4 v4_ = (float4)0;

    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    i1_ = 0;
    i2_ = (0).xx;
    v1_ = 0.0;
    v4_ = (0.0).xxxx;
    float2 uv = (0.0).xx;
    int2 pix = (0).xx;
    int2 _expr23 = i2_;
    i2_ = (_expr23 + NagaDimensions2D(texture_array_unbounded[0]));
    int2 _expr28 = i2_;
    i2_ = (_expr28 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    int2 _expr33 = i2_;
    i2_ = (_expr33 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr41 = texture_array_bounded[0].Gather(samp[0], uv);
    float4 _expr42 = v4_;
    v4_ = (_expr42 + _expr41);
    float4 _expr48 = texture_array_bounded[uniform_index].Gather(samp[uniform_index], uv);
    float4 _expr49 = v4_;
    v4_ = (_expr49 + _expr48);
    float4 _expr55 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr56 = v4_;
    v4_ = (_expr56 + _expr55);
    float4 _expr65 = texture_array_depth[0].GatherCmp(samp_comp[0], uv, 0.0);
    float4 _expr66 = v4_;
    v4_ = (_expr66 + _expr65);
    float4 _expr73 = texture_array_depth[uniform_index].GatherCmp(samp_comp[uniform_index], uv, 0.0);
    float4 _expr74 = v4_;
    v4_ = (_expr74 + _expr73);
    float4 _expr81 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr82 = v4_;
    v4_ = (_expr82 + _expr81);
    float4 _expr88 = texture_array_unbounded[0].Load(int3(pix, 0));
    float4 _expr89 = v4_;
    v4_ = (_expr89 + _expr88);
    float4 _expr94 = texture_array_unbounded[uniform_index].Load(int3(pix, 0));
    float4 _expr95 = v4_;
    v4_ = (_expr95 + _expr94);
    float4 _expr100 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, 0));
    float4 _expr101 = v4_;
    v4_ = (_expr101 + _expr100);
    int _expr107 = i1_;
    i1_ = (_expr107 + NagaNumLayers2DArray(texture_array_2darray[0]));
    int _expr112 = i1_;
    i1_ = (_expr112 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    int _expr117 = i1_;
    i1_ = (_expr117 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    int _expr123 = i1_;
    i1_ = (_expr123 + NagaNumLevels2D(texture_array_bounded[0]));
    int _expr128 = i1_;
    i1_ = (_expr128 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    int _expr133 = i1_;
    i1_ = (_expr133 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    int _expr139 = i1_;
    i1_ = (_expr139 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    int _expr144 = i1_;
    i1_ = (_expr144 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    int _expr149 = i1_;
    i1_ = (_expr149 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr157 = texture_array_bounded[0].Sample(samp[0], uv);
    float4 _expr158 = v4_;
    v4_ = (_expr158 + _expr157);
    float4 _expr164 = texture_array_bounded[uniform_index].Sample(samp[uniform_index], uv);
    float4 _expr165 = v4_;
    v4_ = (_expr165 + _expr164);
    float4 _expr171 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr172 = v4_;
    v4_ = (_expr172 + _expr171);
    float4 _expr181 = texture_array_bounded[0].SampleBias(samp[0], uv, 0.0);
    float4 _expr182 = v4_;
    v4_ = (_expr182 + _expr181);
    float4 _expr189 = texture_array_bounded[uniform_index].SampleBias(samp[uniform_index], uv, 0.0);
    float4 _expr190 = v4_;
    v4_ = (_expr190 + _expr189);
    float4 _expr197 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr198 = v4_;
    v4_ = (_expr198 + _expr197);
    float _expr207 = texture_array_depth[0].SampleCmp(samp_comp[0], uv, 0.0);
    float _expr208 = v1_;
    v1_ = (_expr208 + _expr207);
    float _expr215 = texture_array_depth[uniform_index].SampleCmp(samp_comp[uniform_index], uv, 0.0);
    float _expr216 = v1_;
    v1_ = (_expr216 + _expr215);
    float _expr223 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr224 = v1_;
    v1_ = (_expr224 + _expr223);
    float _expr233 = texture_array_depth[0].SampleCmpLevelZero(samp_comp[0], uv, 0.0);
    float _expr234 = v1_;
    v1_ = (_expr234 + _expr233);
    float _expr241 = texture_array_depth[uniform_index].SampleCmpLevelZero(samp_comp[uniform_index], uv, 0.0);
    float _expr242 = v1_;
    v1_ = (_expr242 + _expr241);
    float _expr249 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr250 = v1_;
    v1_ = (_expr250 + _expr249);
    float4 _expr258 = texture_array_bounded[0].SampleGrad(samp[0], uv, uv, uv);
    float4 _expr259 = v4_;
    v4_ = (_expr259 + _expr258);
    float4 _expr265 = texture_array_bounded[uniform_index].SampleGrad(samp[uniform_index], uv, uv, uv);
    float4 _expr266 = v4_;
    v4_ = (_expr266 + _expr265);
    float4 _expr272 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(samp[NonUniformResourceIndex(non_uniform_index)], uv, uv, uv);
    float4 _expr273 = v4_;
    v4_ = (_expr273 + _expr272);
    float4 _expr282 = texture_array_bounded[0].SampleLevel(samp[0], uv, 0.0);
    float4 _expr283 = v4_;
    v4_ = (_expr283 + _expr282);
    float4 _expr290 = texture_array_bounded[uniform_index].SampleLevel(samp[uniform_index], uv, 0.0);
    float4 _expr291 = v4_;
    v4_ = (_expr291 + _expr290);
    float4 _expr298 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr299 = v4_;
    v4_ = (_expr299 + _expr298);
    float4 _expr304 = v4_;
    texture_array_storage[0][pix] = _expr304;
    float4 _expr307 = v4_;
    texture_array_storage[uniform_index][pix] = _expr307;
    float4 _expr310 = v4_;
    texture_array_storage[NonUniformResourceIndex(non_uniform_index)][pix] = _expr310;
    int2 _expr311 = i2_;
    int _expr312 = i1_;
    float2 v2_ = float2((_expr311 + (_expr312).xx));
    float4 _expr316 = v4_;
    float _expr323 = v1_;
    return ((_expr316 + float4(v2_.x, v2_.y, v2_.x, v2_.y)) + (_expr323).xxxx);
}
