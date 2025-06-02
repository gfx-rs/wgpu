struct NagaExternalTextureParams {
    row_major float4x4 yuv_conversion_matrix;
    float2 sample_transform_0; float2 sample_transform_1; float2 sample_transform_2;
    float2 load_transform_0; float2 load_transform_1; float2 load_transform_2;
    uint2 size;
    uint num_planes;
    int _end_pad_0;
};

Texture2D<float4> tex_plane0_: register(t0);
Texture2D<float4> tex_plane1_: register(t1);
Texture2D<float4> tex_plane2_: register(t2);
cbuffer tex_params: register(b3) { NagaExternalTextureParams tex_params; };
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerState samp = nagaSamplerHeap[nagaGroup0SamplerIndexArray[0]];

float4 nagaTextureSampleBaseClampToEdge(
    Texture2D<float4> plane0,
    Texture2D<float4> plane1,
    Texture2D<float4> plane2,
    NagaExternalTextureParams params,
    SamplerState samp,
    float2 coords)
{
    float2 plane0_size;
    plane0.GetDimensions(plane0_size.x, plane0_size.y);
    float3x2 sample_transform = float3x2(
        params.sample_transform_0,
        params.sample_transform_1,
        params.sample_transform_2
    );
    coords = mul(float3(coords, 1.0), sample_transform);
    float2 bounds_min = mul(float3(0.0, 0.0, 1.0), sample_transform);
    float2 bounds_max = mul(float3(1.0, 1.0, 1.0), sample_transform);
    float4 bounds = float4(min(bounds_min, bounds_max), max(bounds_min, bounds_max));
    float2 plane0_half_texel = float2(0.5, 0.5) / plane0_size;
    float2 plane0_coords = clamp(coords, bounds.xy + plane0_half_texel, bounds.zw - plane0_half_texel);
    if (params.num_planes == 1u) {
        return plane0.SampleLevel(samp, plane0_coords, 0.0f);
    } else {
        float2 plane1_size;
        plane1.GetDimensions(plane1_size.x, plane1_size.y);
        float2 plane1_half_texel = float2(0.5, 0.5) / plane1_size;
        float2 plane1_coords = clamp(coords, bounds.xy + plane1_half_texel, bounds.zw - plane1_half_texel);
        float y = plane0.SampleLevel(samp, plane0_coords, 0.0f).x;
        float2 uv;
        if (params.num_planes == 2u) {
            uv = plane1.SampleLevel(samp, plane1_coords, 0.0f).xy;
        } else {
            float2 plane2_size;
            plane2.GetDimensions(plane2_size.x, plane2_size.y);
            float2 plane2_half_texel = float2(0.5, 0.5) / plane2_size;
            float2 plane2_coords = clamp(coords, bounds.xy + plane2_half_texel, bounds.zw - plane2_half_texel);
            uv = float2(plane1.SampleLevel(samp, plane1_coords, 0.0f).x, plane2.SampleLevel(samp, plane2_coords, 0.0f).x);
        }
        return mul(float4(y, uv, 1.0), params.yuv_conversion_matrix);
    }
}

float4 nagaTextureLoadExternal(
    Texture2D<float4> plane0,
    Texture2D<float4> plane1,
    Texture2D<float4> plane2,
    NagaExternalTextureParams params,
    uint2 coords)
{
    uint2 plane0_size;
    plane0.GetDimensions(plane0_size.x, plane0_size.y);
    uint2 cropped_size = any(params.size) ? params.size : plane0_size;
    coords = min(coords, cropped_size - 1);
    float3x2 load_transform = float3x2(
        params.load_transform_0,
        params.load_transform_1,
        params.load_transform_2
    );
    uint2 plane0_coords = uint2(round(mul(float3(coords, 1.0), load_transform)));
    if (params.num_planes == 1u) {
        return plane0.Load(uint3(plane0_coords, 0u));
    } else {
        uint2 plane1_size;
        plane1.GetDimensions(plane1_size.x, plane1_size.y);
        uint2 plane1_coords = uint2(floor(float2(plane0_coords) * float2(plane1_size) / float2(plane0_size)));
        float y = plane0.Load(uint3(plane0_coords, 0u)).x;
        float2 uv;
        if (params.num_planes == 2u) {
            uv = plane1.Load(uint3(plane1_coords, 0u)).xy;
        } else {
            uint2 plane2_size;
            plane2.GetDimensions(plane2_size.x, plane2_size.y);
            uint2 plane2_coords = uint2(floor(float2(plane0_coords) * float2(plane2_size) / float2(plane0_size)));
            uv = float2(plane1.Load(uint3(plane1_coords, 0u)).x, plane2.Load(uint3(plane2_coords, 0u)).x);
        }
        return mul(float4(y, uv, 1.0), params.yuv_conversion_matrix);
    }
}

uint2 NagaExternalDimensions2D(Texture2D<float4> plane0, Texture2D<float4> plane1, Texture2D<float4> plane2, NagaExternalTextureParams params) {
    if (any(params.size)) {
        return params.size;
    } else {
        uint2 ret;
        plane0.GetDimensions(ret.x, ret.y);
        return ret;
    }
}

float4 test(Texture2D<float4> t_plane0_, Texture2D<float4> t_plane1_, Texture2D<float4> t_plane2_, NagaExternalTextureParams t_params)
{
    float4 a = (float4)0;
    float4 b = (float4)0;
    uint2 c = (uint2)0;

    float4 _e4 = nagaTextureSampleBaseClampToEdge(t_plane0_, t_plane1_, t_plane2_, t_params, samp, (0.0).xx);
    a = _e4;
    float4 _e8 = nagaTextureLoadExternal(t_plane0_, t_plane1_, t_plane2_, t_params, (0u).xx);
    b = _e8;
    c = NagaExternalDimensions2D(t_plane0_, t_plane1_, t_plane2_, t_params);
    float4 _e12 = a;
    float4 _e13 = b;
    uint2 _e15 = c;
    return ((_e12 + _e13) + float2(_e15).xyxy);
}

float4 fragment_main() : SV_Target0
{
    const float4 _e1 = test(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    return _e1;
}

float4 vertex_main() : SV_Position
{
    const float4 _e1 = test(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    return _e1;
}

[numthreads(1, 1, 1)]
void compute_main()
{
    const float4 _e1 = test(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    return;
}
