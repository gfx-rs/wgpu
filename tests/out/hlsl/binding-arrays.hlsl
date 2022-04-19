
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
    int i1_ = 0;
    int2 i2_ = (int2)0;
    float v1_ = 0.0;
    float4 v4_ = (float4)0;

    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    i2_ = (0).xx;
    v4_ = (0.0).xxxx;
    float2 uv = (0.0).xx;
    int2 pix = (0).xx;
    int2 _expr27 = i2_;
    i2_ = (_expr27 + NagaDimensions2D(texture_array_unbounded[0]));
    int2 _expr32 = i2_;
    i2_ = (_expr32 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    int2 _expr36 = i2_;
    i2_ = (_expr36 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr40 = v4_;
    float4 _expr45 = texture_array_bounded[0].Gather(samp[0], uv);
    v4_ = (_expr40 + _expr45);
    float4 _expr47 = v4_;
    float4 _expr50 = texture_array_bounded[uniform_index].Gather(samp[uniform_index], uv);
    v4_ = (_expr47 + _expr50);
    float4 _expr52 = v4_;
    float4 _expr55 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    v4_ = (_expr52 + _expr55);
    float4 _expr57 = v4_;
    float4 _expr63 = texture_array_depth[0].GatherCmp(samp_comp[0], uv, 0.0);
    v4_ = (_expr57 + _expr63);
    float4 _expr65 = v4_;
    float4 _expr69 = texture_array_depth[uniform_index].GatherCmp(samp_comp[uniform_index], uv, 0.0);
    v4_ = (_expr65 + _expr69);
    float4 _expr71 = v4_;
    float4 _expr75 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    v4_ = (_expr71 + _expr75);
    float4 _expr77 = v4_;
    float4 _expr81 = texture_array_unbounded[0].Load(int3(pix, 0));
    v4_ = (_expr77 + _expr81);
    float4 _expr83 = v4_;
    float4 _expr86 = texture_array_unbounded[uniform_index].Load(int3(pix, 0));
    v4_ = (_expr83 + _expr86);
    float4 _expr88 = v4_;
    float4 _expr91 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, 0));
    v4_ = (_expr88 + _expr91);
    int _expr93 = i1_;
    i1_ = (_expr93 + NagaNumLayers2DArray(texture_array_2darray[0]));
    int _expr98 = i1_;
    i1_ = (_expr98 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    int _expr102 = i1_;
    i1_ = (_expr102 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    int _expr106 = i1_;
    i1_ = (_expr106 + NagaNumLevels2D(texture_array_bounded[0]));
    int _expr111 = i1_;
    i1_ = (_expr111 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    int _expr115 = i1_;
    i1_ = (_expr115 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    int _expr119 = i1_;
    i1_ = (_expr119 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    int _expr124 = i1_;
    i1_ = (_expr124 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    int _expr128 = i1_;
    i1_ = (_expr128 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr132 = v4_;
    float4 _expr137 = texture_array_bounded[0].Sample(samp[0], uv);
    v4_ = (_expr132 + _expr137);
    float4 _expr139 = v4_;
    float4 _expr142 = texture_array_bounded[uniform_index].Sample(samp[uniform_index], uv);
    v4_ = (_expr139 + _expr142);
    float4 _expr144 = v4_;
    float4 _expr147 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    v4_ = (_expr144 + _expr147);
    float4 _expr149 = v4_;
    float4 _expr155 = texture_array_bounded[0].SampleBias(samp[0], uv, 0.0);
    v4_ = (_expr149 + _expr155);
    float4 _expr157 = v4_;
    float4 _expr161 = texture_array_bounded[uniform_index].SampleBias(samp[uniform_index], uv, 0.0);
    v4_ = (_expr157 + _expr161);
    float4 _expr163 = v4_;
    float4 _expr167 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    v4_ = (_expr163 + _expr167);
    float _expr169 = v1_;
    float _expr175 = texture_array_depth[0].SampleCmp(samp_comp[0], uv, 0.0);
    v1_ = (_expr169 + _expr175);
    float _expr177 = v1_;
    float _expr181 = texture_array_depth[uniform_index].SampleCmp(samp_comp[uniform_index], uv, 0.0);
    v1_ = (_expr177 + _expr181);
    float _expr183 = v1_;
    float _expr187 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    v1_ = (_expr183 + _expr187);
    float _expr189 = v1_;
    float _expr195 = texture_array_depth[0].SampleCmpLevelZero(samp_comp[0], uv, 0.0);
    v1_ = (_expr189 + _expr195);
    float _expr197 = v1_;
    float _expr201 = texture_array_depth[uniform_index].SampleCmpLevelZero(samp_comp[uniform_index], uv, 0.0);
    v1_ = (_expr197 + _expr201);
    float _expr203 = v1_;
    float _expr207 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    v1_ = (_expr203 + _expr207);
    float4 _expr209 = v4_;
    float4 _expr214 = texture_array_bounded[0].SampleGrad(samp[0], uv, uv, uv);
    v4_ = (_expr209 + _expr214);
    float4 _expr216 = v4_;
    float4 _expr219 = texture_array_bounded[uniform_index].SampleGrad(samp[uniform_index], uv, uv, uv);
    v4_ = (_expr216 + _expr219);
    float4 _expr221 = v4_;
    float4 _expr224 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(samp[NonUniformResourceIndex(non_uniform_index)], uv, uv, uv);
    v4_ = (_expr221 + _expr224);
    float4 _expr226 = v4_;
    float4 _expr232 = texture_array_bounded[0].SampleLevel(samp[0], uv, 0.0);
    v4_ = (_expr226 + _expr232);
    float4 _expr234 = v4_;
    float4 _expr238 = texture_array_bounded[uniform_index].SampleLevel(samp[uniform_index], uv, 0.0);
    v4_ = (_expr234 + _expr238);
    float4 _expr240 = v4_;
    float4 _expr244 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    v4_ = (_expr240 + _expr244);
    float4 _expr248 = v4_;
    texture_array_storage[0][pix] = _expr248;
    float4 _expr250 = v4_;
    texture_array_storage[uniform_index][pix] = _expr250;
    float4 _expr252 = v4_;
    texture_array_storage[NonUniformResourceIndex(non_uniform_index)][pix] = _expr252;
    int2 _expr253 = i2_;
    int _expr254 = i1_;
    float2 v2_ = float2((_expr253 + (_expr254).xx));
    float4 _expr258 = v4_;
    float _expr265 = v1_;
    return ((_expr258 + float4(v2_.x, v2_.y, v2_.x, v2_.y)) + (_expr265).xxxx);
}
