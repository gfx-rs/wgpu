
Texture2D<uint4> image_mipmapped_src : register(t0);
Texture2DMS<uint4> image_multisampled_src : register(t3);
Texture2DMS<float> image_depth_multisampled_src : register(t4);
Texture2D<uint4> image_storage_src : register(t1);
RWTexture1D<uint4> image_dst : register(u2);
Texture1D<float4> image_1d : register(t0);
Texture2D<float4> image_2d : register(t1);
Texture2DArray<float4> image_2d_array : register(t2);
TextureCube<float4> image_cube : register(t3);
TextureCubeArray<float4> image_cube_array : register(t4);
Texture3D<float4> image_3d : register(t5);
Texture2DMS<float4> image_aa : register(t6);
SamplerState sampler_reg : register(s0);
SamplerComparisonState sampler_cmp : register(s1);
Texture2D<float> image_2d_depth : register(t2);

struct ComputeInput_main {
    uint3 local_id1 : SV_GroupThreadID;
};

int2 NagaDimensions2D(Texture2D<uint4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

[numthreads(16, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    int2 dim = NagaDimensions2D(image_storage_src);
    int2 itc = ((dim * int2(computeinput_main.local_id1.xy)) % int2(10, 20));
    uint4 value1_ = image_mipmapped_src.Load(int3(itc, int(computeinput_main.local_id1.z)));
    uint4 value2_ = image_multisampled_src.Load(itc, int(computeinput_main.local_id1.z));
    float value3_ = image_depth_multisampled_src.Load(itc, int(computeinput_main.local_id1.z)).x;
    uint4 value4_ = image_storage_src.Load(int3(itc, 0));
    image_dst[itc.x] = (((value1_ + value2_) + uint4(uint(value3_).xxxx)) + value4_);
    return;
}

int NagaDimensions1D(Texture1D<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(ret.x);
    return ret.x;
}

int2 NagaDimensions2D(Texture2D<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

int NagaNumLevels2D(Texture2D<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

int2 NagaMipDimensions2D(Texture2D<float4> texture, uint mip_level)
{
    uint4 ret;
    texture.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaDimensions2DArray(Texture2DArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int NagaNumLevels2DArray(Texture2DArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int2 NagaMipDimensions2DArray(Texture2DArray<float4> texture, uint mip_level)
{
    uint4 ret;
    texture.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int NagaNumLayers2DArray(Texture2DArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int2 NagaDimensionsCube(TextureCube<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

int NagaNumLevelsCube(TextureCube<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

int2 NagaMipDimensionsCube(TextureCube<float4> texture, uint mip_level)
{
    uint4 ret;
    texture.GetDimensions(mip_level, ret.x, ret.y, ret.z);
    return ret.xy;
}

int2 NagaDimensionsCubeArray(TextureCubeArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int NagaNumLevelsCubeArray(TextureCubeArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int2 NagaMipDimensionsCubeArray(TextureCubeArray<float4> texture, uint mip_level)
{
    uint4 ret;
    texture.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xy;
}

int NagaNumLayersCubeArray(TextureCubeArray<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int3 NagaDimensions3D(Texture3D<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

int NagaNumLevels3D(Texture3D<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

int3 NagaMipDimensions3D(Texture3D<float4> texture, uint mip_level)
{
    uint4 ret;
    texture.GetDimensions(mip_level, ret.x, ret.y, ret.z, ret.w);
    return ret.xyz;
}

int NagaMSNumSamples2D(Texture2DMS<float4> texture)
{
    uint4 ret;
    texture.GetDimensions(ret.x, ret.y, ret.z);
    return ret.z;
}

float4 queries() : SV_Position
{
    int dim_1d = NagaDimensions1D(image_1d);
    int2 dim_2d = NagaDimensions2D(image_2d);
    int num_levels_2d = NagaNumLevels2D(image_2d);
    int2 dim_2d_lod = NagaMipDimensions2D(image_2d, 1);
    int2 dim_2d_array = NagaDimensions2DArray(image_2d_array);
    int num_levels_2d_array = NagaNumLevels2DArray(image_2d_array);
    int2 dim_2d_array_lod = NagaMipDimensions2DArray(image_2d_array, 1);
    int num_layers_2d = NagaNumLayers2DArray(image_2d_array);
    int2 dim_cube = NagaDimensionsCube(image_cube);
    int num_levels_cube = NagaNumLevelsCube(image_cube);
    int2 dim_cube_lod = NagaMipDimensionsCube(image_cube, 1);
    int2 dim_cube_array = NagaDimensionsCubeArray(image_cube_array);
    int num_levels_cube_array = NagaNumLevelsCubeArray(image_cube_array);
    int2 dim_cube_array_lod = NagaMipDimensionsCubeArray(image_cube_array, 1);
    int num_layers_cube = NagaNumLayersCubeArray(image_cube_array);
    int3 dim_3d = NagaDimensions3D(image_3d);
    int num_levels_3d = NagaNumLevels3D(image_3d);
    int3 dim_3d_lod = NagaMipDimensions3D(image_3d, 1);
    int num_samples_aa = NagaMSNumSamples2D(image_aa);
    int sum = ((((((((((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + num_layers_2d) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + num_layers_cube) + dim_3d.z) + dim_3d_lod.z) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return float4(float(sum).xxxx);
}

float4 sample1() : SV_Target0
{
    float2 tc = float2(0.5.xx);
    float4 s2d = image_2d.Sample(sampler_reg, tc);
    float4 s2d_offset = image_2d.Sample(sampler_reg, tc, int2(3, 1));
    float4 s2d_level = image_2d.SampleLevel(sampler_reg, tc, 2.3);
    float4 s2d_level_offset = image_2d.SampleLevel(sampler_reg, tc, 2.3, int2(3, 1));
    return (((s2d + s2d_offset) + s2d_level) + s2d_level_offset);
}

float sample_comparison() : SV_Target0
{
    float2 tc = float2(0.5.xx);
    float s2d_depth = image_2d_depth.SampleCmp(sampler_cmp, tc, 0.5);
    float s2d_depth_level = image_2d_depth.SampleCmpLevelZero(sampler_cmp, tc, 0.5);
    return (s2d_depth + s2d_depth_level);
}
