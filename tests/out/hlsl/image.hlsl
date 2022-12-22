
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
TextureCube<float> image_cube_depth : register(t3, space1);

int2 NagaRWDimensions2D(RWTexture2D<uint4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y);
    return ret.xy;
}

[numthreads(16, 1, 1)]
void main(uint3 local_id : SV_GroupThreadID)
{
    int2 dim = NagaRWDimensions2D(image_storage_src);
    int2 itc = ((dim * int2(local_id.xy)) % int2(10, 20));
    uint4 value1_ = image_mipmapped_src.Load(int3(itc, int(local_id.z)));
    uint4 value2_ = image_multisampled_src.Load(itc, int(local_id.z));
    uint4 value4_ = image_storage_src.Load(itc);
    uint4 value5_ = image_array_src.Load(int4(itc, int(local_id.z), (int(local_id.z) + 1)));
    uint4 value6_ = image_1d_src.Load(int2(int(local_id.x), int(local_id.z)));
    uint4 value1u = image_mipmapped_src.Load(int3(uint2(itc), int(local_id.z)));
    uint4 value2u = image_multisampled_src.Load(uint2(itc), int(local_id.z));
    uint4 value4u = image_storage_src.Load(uint2(itc));
    uint4 value5u = image_array_src.Load(int4(uint2(itc), int(local_id.z), (int(local_id.z) + 1)));
    uint4 value6u = image_1d_src.Load(int2(uint(local_id.x), int(local_id.z)));
    image_dst[itc.x] = ((((value1_ + value2_) + value4_) + value5_) + value6_);
    image_dst[uint(itc.x)] = ((((value1u + value2u) + value4u) + value5u) + value6u);
    return;
}

[numthreads(16, 1, 1)]
void depth_load(uint3 local_id_1 : SV_GroupThreadID)
{
    int2 dim_1 = NagaRWDimensions2D(image_storage_src);
    int2 itc_1 = ((dim_1 * int2(local_id_1.xy)) % int2(10, 20));
    float val = image_depth_multisampled_src.Load(itc_1, int(local_id_1.z)).x;
    image_dst[itc_1.x] = (uint(val)).xxxx;
    return;
}

int NagaDimensions1D(Texture1D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y);
    return ret.x;
}

int NagaMipDimensions1D(Texture1D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y);
    return ret.x;
}

int2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaMipDimensions2D(Texture2D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaDimensions2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int2 NagaMipDimensions2DArray(Texture2DArray<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int2 NagaDimensionsCube(TextureCube<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaMipDimensionsCube(TextureCube<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaDimensionsCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int2 NagaMipDimensionsCubeArray(TextureCubeArray<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int3 NagaDimensions3D(Texture3D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

int3 NagaMipDimensions3D(Texture3D<float4> tex, uint mip_level)
{
    uint4 ret;
    tex.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

int2 NagaMSDimensions2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.xy;
}

float4 queries() : SV_Position
{
    int dim_1d = NagaDimensions1D(image_1d);
    int dim_1d_lod = NagaMipDimensions1D(image_1d, int(dim_1d));
    int2 dim_2d = NagaDimensions2D(image_2d);
    int2 dim_2d_lod = NagaMipDimensions2D(image_2d, 1);
    int2 dim_2d_array = NagaDimensions2DArray(image_2d_array);
    int2 dim_2d_array_lod = NagaMipDimensions2DArray(image_2d_array, 1);
    int2 dim_cube = NagaDimensionsCube(image_cube);
    int2 dim_cube_lod = NagaMipDimensionsCube(image_cube, 1);
    int2 dim_cube_array = NagaDimensionsCubeArray(image_cube_array);
    int2 dim_cube_array_lod = NagaMipDimensionsCubeArray(image_cube_array, 1);
    int3 dim_3d = NagaDimensions3D(image_3d);
    int3 dim_3d_lod = NagaMipDimensions3D(image_3d, 1);
    int2 dim_2s_ms = NagaMSDimensions2D(image_aa);
    int sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return (float(sum)).xxxx;
}

int NagaNumLevels2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

int NagaNumLevels2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaNumLayers2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaNumLevelsCube(TextureCube<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

int NagaNumLevelsCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaNumLayersCubeArray(TextureCubeArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaNumLevels3D(Texture3D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int NagaMSNumSamples2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.z;
}

float4 levels_queries() : SV_Position
{
    int num_levels_2d = NagaNumLevels2D(image_2d);
    int num_levels_2d_array = NagaNumLevels2DArray(image_2d_array);
    int num_layers_2d = NagaNumLayers2DArray(image_2d_array);
    int num_levels_cube = NagaNumLevelsCube(image_cube);
    int num_levels_cube_array = NagaNumLevelsCubeArray(image_cube_array);
    int num_layers_cube = NagaNumLayersCubeArray(image_cube_array);
    int num_levels_3d = NagaNumLevels3D(image_3d);
    int num_samples_aa = NagaMSNumSamples2D(image_aa);
    int sum_1 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return (float(sum_1)).xxxx;
}

float4 texture_sample() : SV_Target0
{
    float2 tc = (0.5).xx;
    float4 s1d = image_1d.Sample(sampler_reg, tc.x);
    float4 s2d = image_2d.Sample(sampler_reg, tc);
    float4 s2d_offset = image_2d.Sample(sampler_reg, tc, int2(3, 1));
    float4 s2d_level = image_2d.SampleLevel(sampler_reg, tc, 2.299999952316284);
    float4 s2d_level_offset = image_2d.SampleLevel(sampler_reg, tc, 2.299999952316284, int2(3, 1));
    float4 s2d_bias_offset = image_2d.SampleBias(sampler_reg, tc, 2.0, int2(3, 1));
    return ((((s1d + s2d) + s2d_offset) + s2d_level) + s2d_level_offset);
}

float texture_sample_comparison() : SV_Target0
{
    float2 tc_1 = (0.5).xx;
    float s2d_depth = image_2d_depth.SampleCmp(sampler_cmp, tc_1, 0.5);
    float s2d_depth_level = image_2d_depth.SampleCmpLevelZero(sampler_cmp, tc_1, 0.5);
    float scube_depth_level = image_cube_depth.SampleCmpLevelZero(sampler_cmp, (0.5).xxx, 0.5);
    return (s2d_depth + s2d_depth_level);
}

float4 gather() : SV_Target0
{
    float2 tc_2 = (0.5).xx;
    float4 s2d_1 = image_2d.GatherGreen(sampler_reg, tc_2);
    float4 s2d_offset_1 = image_2d.GatherAlpha(sampler_reg, tc_2, int2(3, 1));
    float4 s2d_depth_1 = image_2d_depth.GatherCmp(sampler_cmp, tc_2, 0.5);
    float4 s2d_depth_offset = image_2d_depth.GatherCmp(sampler_cmp, tc_2, 0.5, int2(3, 1));
    uint4 u = image_2d_u32_.Gather(sampler_reg, tc_2);
    int4 i = image_2d_i32_.Gather(sampler_reg, tc_2);
    float4 f = (float4(u) + float4(i));
    return ((((s2d_1 + s2d_offset_1) + s2d_depth_1) + s2d_depth_offset) + f);
}

float4 depth_no_comparison() : SV_Target0
{
    float2 tc_3 = (0.5).xx;
    float s2d_2 = image_2d_depth.Sample(sampler_reg, tc_3);
    float4 s2d_gather = image_2d_depth.Gather(sampler_reg, tc_3);
    return ((s2d_2).xxxx + s2d_gather);
}
