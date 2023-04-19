Texture2D<uint4> image_mipmapped_src : register(t0);
Texture2DMS<uint4> image_multisampled_src : register(t3);
Texture2DMS<float> image_depth_multisampled_src : register(t4);
RWTexture2D<uint4> image_storage_src : register(u1);
Texture2DArray<uint4> image_array_src : register(t5);
RWTexture1D<uint4> image_dup_src : register(u6);
Texture1D<uint4> image_1d_src : register(t7);
RWTexture1D<uint4> image_dst : register(u2);
Texture1D<float4> image_1d : register(t0);
Texture2D<float4> image_2d : register(t1);
Texture2D<uint4> image_2d_u32_ : register(t2);
Texture2D<int4> image_2d_i32_ : register(t3);
Texture2DArray<float4> image_2d_array : register(t4);
TextureCube<float4> image_cube : register(t5);
TextureCubeArray<float4> image_cube_array : register(t6);
Texture3D<float4> image_3d : register(t7);
Texture2DMS<float4> image_aa : register(t8);
SamplerState sampler_reg : register(s0, space1);
SamplerComparisonState sampler_cmp : register(s1, space1);
Texture2D<float> image_2d_depth : register(t2, space1);
Texture2DArray<float> image_2d_array_depth : register(t3, space1);
TextureCube<float> image_cube_depth : register(t4, space1);

uint2 NagaRWDimensions2D(RWTexture2D<uint4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y);
    return ret.xy;
}

[numthreads(16, 1, 1)]
void main(uint3 local_id : SV_GroupThreadID)
{
    uint2 dim = NagaRWDimensions2D(image_storage_src);
    int2 itc = (int2((dim * local_id.xy)) % int2(10, 20));
    uint4 value1_ = image_mipmapped_src.Load(int3(itc, int(local_id.z)));
    uint4 value2_ = image_multisampled_src.Load(itc, int(local_id.z));
    uint4 value4_ = image_storage_src.Load(itc);
    uint4 value5_ = image_array_src.Load(int4(itc, local_id.z, (int(local_id.z) + 1)));
    uint4 value6_ = image_array_src.Load(int4(itc, int(local_id.z), (int(local_id.z) + 1)));
    uint4 value7_ = image_1d_src.Load(int2(int(local_id.x), int(local_id.z)));
    uint4 value1u = image_mipmapped_src.Load(int3(uint2(itc), int(local_id.z)));
    uint4 value2u = image_multisampled_src.Load(uint2(itc), int(local_id.z));
    uint4 value4u = image_storage_src.Load(uint2(itc));
    uint4 value5u = image_array_src.Load(int4(uint2(itc), local_id.z, (int(local_id.z) + 1)));
    uint4 value6u = image_array_src.Load(int4(uint2(itc), int(local_id.z), (int(local_id.z) + 1)));
    uint4 value7u = image_1d_src.Load(int2(uint(local_id.x), int(local_id.z)));
    image_dst[itc.x] = ((((value1_ + value2_) + value4_) + value5_) + value6_);
    image_dst[uint(itc.x)] = ((((value1u + value2u) + value4u) + value5u) + value6u);
    return;
}

[numthreads(16, 1, 1)]
void depth_load(uint3 local_id_1 : SV_GroupThreadID)
{
    uint2 dim_1 = NagaRWDimensions2D(image_storage_src);
    int2 itc_1 = (int2((dim_1 * local_id_1.xy)) % int2(10, 20));
    float val = image_depth_multisampled_src.Load(itc_1, int(local_id_1.z)).x;
    image_dst[itc_1.x] = (uint(val)).xxxx;
    return;
}

uint NagaDimensions1D(Texture1D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y);
    return ret.x;
}

uint NagaMipDimensions1D(Texture1D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y);
    return ret.x;
}

uint2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

uint2 NagaMipDimensions2D(Texture2D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

uint2 NagaDimensions2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

uint2 NagaMipDimensions2DArray(Texture2DArray<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

uint2 NagaDimensionsCube(TextureCube<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

uint2 NagaMipDimensionsCube(TextureCube<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

uint2 NagaDimensionsCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

uint2 NagaMipDimensionsCubeArray(TextureCubeArray<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

uint3 NagaDimensions3D(Texture3D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

uint3 NagaMipDimensions3D(Texture3D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

uint2 NagaMSDimensions2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.xy;
}

float4 queries() : SV_Position
{
    uint dim_1d = NagaDimensions1D(image_1d);
    uint dim_1d_lod = NagaMipDimensions1D(image_1d, int(dim_1d));
    uint2 dim_2d = NagaDimensions2D(image_2d);
    uint2 dim_2d_lod = NagaMipDimensions2D(image_2d, 1);
    uint2 dim_2d_array = NagaDimensions2DArray(image_2d_array);
    uint2 dim_2d_array_lod = NagaMipDimensions2DArray(image_2d_array, 1);
    uint2 dim_cube = NagaDimensionsCube(image_cube);
    uint2 dim_cube_lod = NagaMipDimensionsCube(image_cube, 1);
    uint2 dim_cube_array = NagaDimensionsCubeArray(image_cube_array);
    uint2 dim_cube_array_lod = NagaMipDimensionsCubeArray(image_cube_array, 1);
    uint3 dim_3d = NagaDimensions3D(image_3d);
    uint3 dim_3d_lod = NagaMipDimensions3D(image_3d, 1);
    uint2 dim_2s_ms = NagaMSDimensions2D(image_aa);
    uint sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return (float(sum)).xxxx;
}

uint NagaNumLevels2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

uint NagaNumLevels2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaNumLayers2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaNumLevelsCube(TextureCube<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

uint NagaNumLevelsCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaNumLayersCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaNumLevels3D(Texture3D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaMSNumSamples2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.z;
}

float4 levels_queries() : SV_Position
{
    uint num_levels_2d = NagaNumLevels2D(image_2d);
    uint num_levels_2d_array = NagaNumLevels2DArray(image_2d_array);
    uint num_layers_2d = NagaNumLayers2DArray(image_2d_array);
    uint num_levels_cube = NagaNumLevelsCube(image_cube);
    uint num_levels_cube_array = NagaNumLevelsCubeArray(image_cube_array);
    uint num_layers_cube = NagaNumLayersCubeArray(image_cube_array);
    uint num_levels_3d = NagaNumLevels3D(image_3d);
    uint num_samples_aa = NagaMSNumSamples2D(image_aa);
    uint sum_1 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return (float(sum_1)).xxxx;
}

float4 texture_sample() : SV_Target0
{
    float4 a = (float4)0;

    float2 tc = (0.5).xx;
    float3 tc3_ = (0.5).xxx;
    float4 _expr8 = image_1d.Sample(sampler_reg, 0.5);
    float4 _expr9 = a;
    a = (_expr9 + _expr8);
    float4 _expr13 = image_2d.Sample(sampler_reg, tc);
    float4 _expr14 = a;
    a = (_expr14 + _expr13);
    float4 _expr18 = image_2d.Sample(sampler_reg, tc, int2(int2(3, 1)));
    float4 _expr19 = a;
    a = (_expr19 + _expr18);
    float4 _expr23 = image_2d.SampleLevel(sampler_reg, tc, 2.3);
    float4 _expr24 = a;
    a = (_expr24 + _expr23);
    float4 _expr28 = image_2d.SampleLevel(sampler_reg, tc, 2.3, int2(int2(3, 1)));
    float4 _expr29 = a;
    a = (_expr29 + _expr28);
    float4 _expr34 = image_2d.SampleBias(sampler_reg, tc, 2.0, int2(int2(3, 1)));
    float4 _expr35 = a;
    a = (_expr35 + _expr34);
    float4 _expr40 = image_2d_array.Sample(sampler_reg, float3(tc, 0u));
    float4 _expr41 = a;
    a = (_expr41 + _expr40);
    float4 _expr46 = image_2d_array.Sample(sampler_reg, float3(tc, 0u), int2(int2(3, 1)));
    float4 _expr47 = a;
    a = (_expr47 + _expr46);
    float4 _expr52 = image_2d_array.SampleLevel(sampler_reg, float3(tc, 0u), 2.3);
    float4 _expr53 = a;
    a = (_expr53 + _expr52);
    float4 _expr58 = image_2d_array.SampleLevel(sampler_reg, float3(tc, 0u), 2.3, int2(int2(3, 1)));
    float4 _expr59 = a;
    a = (_expr59 + _expr58);
    float4 _expr65 = image_2d_array.SampleBias(sampler_reg, float3(tc, 0u), 2.0, int2(int2(3, 1)));
    float4 _expr66 = a;
    a = (_expr66 + _expr65);
    float4 _expr71 = image_2d_array.Sample(sampler_reg, float3(tc, 0));
    float4 _expr72 = a;
    a = (_expr72 + _expr71);
    float4 _expr77 = image_2d_array.Sample(sampler_reg, float3(tc, 0), int2(int2(3, 1)));
    float4 _expr78 = a;
    a = (_expr78 + _expr77);
    float4 _expr83 = image_2d_array.SampleLevel(sampler_reg, float3(tc, 0), 2.3);
    float4 _expr84 = a;
    a = (_expr84 + _expr83);
    float4 _expr89 = image_2d_array.SampleLevel(sampler_reg, float3(tc, 0), 2.3, int2(int2(3, 1)));
    float4 _expr90 = a;
    a = (_expr90 + _expr89);
    float4 _expr96 = image_2d_array.SampleBias(sampler_reg, float3(tc, 0), 2.0, int2(int2(3, 1)));
    float4 _expr97 = a;
    a = (_expr97 + _expr96);
    float4 _expr102 = image_cube_array.Sample(sampler_reg, float4(tc3_, 0u));
    float4 _expr103 = a;
    a = (_expr103 + _expr102);
    float4 _expr108 = image_cube_array.SampleLevel(sampler_reg, float4(tc3_, 0u), 2.3);
    float4 _expr109 = a;
    a = (_expr109 + _expr108);
    float4 _expr115 = image_cube_array.SampleBias(sampler_reg, float4(tc3_, 0u), 2.0);
    float4 _expr116 = a;
    a = (_expr116 + _expr115);
    float4 _expr121 = image_cube_array.Sample(sampler_reg, float4(tc3_, 0));
    float4 _expr122 = a;
    a = (_expr122 + _expr121);
    float4 _expr127 = image_cube_array.SampleLevel(sampler_reg, float4(tc3_, 0), 2.3);
    float4 _expr128 = a;
    a = (_expr128 + _expr127);
    float4 _expr134 = image_cube_array.SampleBias(sampler_reg, float4(tc3_, 0), 2.0);
    float4 _expr135 = a;
    a = (_expr135 + _expr134);
    float4 _expr137 = a;
    return _expr137;
}

float texture_sample_comparison() : SV_Target0
{
    float a_1 = (float)0;

    float2 tc_1 = (0.5).xx;
    float3 tc3_1 = (0.5).xxx;
    float _expr8 = image_2d_depth.SampleCmp(sampler_cmp, tc_1, 0.5);
    float _expr9 = a_1;
    a_1 = (_expr9 + _expr8);
    float _expr14 = image_2d_array_depth.SampleCmp(sampler_cmp, float3(tc_1, 0u), 0.5);
    float _expr15 = a_1;
    a_1 = (_expr15 + _expr14);
    float _expr20 = image_2d_array_depth.SampleCmp(sampler_cmp, float3(tc_1, 0), 0.5);
    float _expr21 = a_1;
    a_1 = (_expr21 + _expr20);
    float _expr25 = image_cube_depth.SampleCmp(sampler_cmp, tc3_1, 0.5);
    float _expr26 = a_1;
    a_1 = (_expr26 + _expr25);
    float _expr30 = image_2d_depth.SampleCmpLevelZero(sampler_cmp, tc_1, 0.5);
    float _expr31 = a_1;
    a_1 = (_expr31 + _expr30);
    float _expr36 = image_2d_array_depth.SampleCmpLevelZero(sampler_cmp, float3(tc_1, 0u), 0.5);
    float _expr37 = a_1;
    a_1 = (_expr37 + _expr36);
    float _expr42 = image_2d_array_depth.SampleCmpLevelZero(sampler_cmp, float3(tc_1, 0), 0.5);
    float _expr43 = a_1;
    a_1 = (_expr43 + _expr42);
    float _expr47 = image_cube_depth.SampleCmpLevelZero(sampler_cmp, tc3_1, 0.5);
    float _expr48 = a_1;
    a_1 = (_expr48 + _expr47);
    float _expr50 = a_1;
    return _expr50;
}

float4 gather() : SV_Target0
{
    float2 tc_2 = (0.5).xx;
    float4 s2d = image_2d.GatherGreen(sampler_reg, tc_2);
    float4 s2d_offset = image_2d.GatherAlpha(sampler_reg, tc_2, int2(int2(3, 1)));
    float4 s2d_depth = image_2d_depth.GatherCmp(sampler_cmp, tc_2, 0.5);
    float4 s2d_depth_offset = image_2d_depth.GatherCmp(sampler_cmp, tc_2, 0.5, int2(int2(3, 1)));
    uint4 u = image_2d_u32_.Gather(sampler_reg, tc_2);
    int4 i = image_2d_i32_.Gather(sampler_reg, tc_2);
    float4 f = (float4(u) + float4(i));
    return ((((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset) + f);
}

float4 depth_no_comparison() : SV_Target0
{
    float2 tc_3 = (0.5).xx;
    float s2d_1 = image_2d_depth.Sample(sampler_reg, tc_3);
    float4 s2d_gather = image_2d_depth.Gather(sampler_reg, tc_3);
    return ((s2d_1).xxxx + s2d_gather);
}
