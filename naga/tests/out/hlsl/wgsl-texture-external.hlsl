struct NagaExternalTextureTransferFn {
    float a;
    float b;
    float g;
    float k;
};

struct NagaExternalTextureParams {
    row_major float4x4 yuv_conversion_matrix;
    row_major float3x3 gamut_conversion_matrix;
    int _pad2_0;
    NagaExternalTextureTransferFn src_tf;
    NagaExternalTextureTransferFn dst_tf;
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

float4 ZeroValuefloat4() {
    return (float4)0;
}

uint2 ZeroValueuint2() {
    return (uint2)0;
}

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
        float3 srcGammaRgb = mul(float4(y, uv, 1.0), params.yuv_conversion_matrix).rgb;
        float3 srcLinearRgb = srcGammaRgb < params.src_tf.k * params.src_tf.b ?
            srcGammaRgb / params.src_tf.k :
            pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, params.src_tf.g);
        float3 dstLinearRgb = mul(srcLinearRgb, params.gamut_conversion_matrix);
        float3 dstGammaRgb = dstLinearRgb < params.dst_tf.b ?
            params.dst_tf.k * dstLinearRgb :
            params.dst_tf.a * pow(dstLinearRgb, 1.0 / params.dst_tf.g) - (params.dst_tf.a - 1);
        return float4(dstGammaRgb, 1.0);
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
        float3 srcGammaRgb = mul(float4(y, uv, 1.0), params.yuv_conversion_matrix).rgb;
        float3 srcLinearRgb = srcGammaRgb < params.src_tf.k * params.src_tf.b ?
            srcGammaRgb / params.src_tf.k :
            pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, params.src_tf.g);
        float3 dstLinearRgb = mul(srcLinearRgb, params.gamut_conversion_matrix);
        float3 dstGammaRgb = dstLinearRgb < params.dst_tf.b ?
            params.dst_tf.k * dstLinearRgb :
            params.dst_tf.a * pow(dstLinearRgb, 1.0 / params.dst_tf.g) - (params.dst_tf.a - 1);
        return float4(dstGammaRgb, 1.0);
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

float4 fragment_main() : SV_Target0
{
    float4 a = (float4)0;
    float4 b = (float4)0;
    float4 c = (float4)0;
    uint2 d = (uint2)0;
    float4 local = (float4)0;

    a = ZeroValuefloat4();
    b = ZeroValuefloat4();
    c = ZeroValuefloat4();
    d = ZeroValueuint2();
    float4 _e4 = nagaTextureSampleBaseClampToEdge(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, samp, (0.0).xx);
    a = _e4;
    float4 _e7 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (int(0)).xx);
    b = _e7;
    float4 _e10 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (0u).xx);
    c = _e10;
    d = NagaExternalDimensions2D(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    float4 _e16 = a;
    float4 _e17 = b;
    float4 _e19 = c;
    uint2 _e21 = d;
    local = (((_e16 + _e17) + _e19) + float2(_e21).xyxy);
    float4 _e26 = local;
    return _e26;
}

float4 vertex_main() : SV_Position
{
    float4 a_1 = (float4)0;
    float4 b_1 = (float4)0;
    float4 c_1 = (float4)0;
    uint2 d_1 = (uint2)0;
    float4 local_1 = (float4)0;

    a_1 = ZeroValuefloat4();
    b_1 = ZeroValuefloat4();
    c_1 = ZeroValuefloat4();
    d_1 = ZeroValueuint2();
    float4 _e4 = nagaTextureSampleBaseClampToEdge(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, samp, (0.0).xx);
    a_1 = _e4;
    float4 _e7 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (int(0)).xx);
    b_1 = _e7;
    float4 _e10 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (0u).xx);
    c_1 = _e10;
    d_1 = NagaExternalDimensions2D(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    float4 _e16 = a_1;
    float4 _e17 = b_1;
    float4 _e19 = c_1;
    uint2 _e21 = d_1;
    local_1 = (((_e16 + _e17) + _e19) + float2(_e21).xyxy);
    float4 _e26 = local_1;
    return _e26;
}

[numthreads(1, 1, 1)]
void compute_main()
{
    float4 a_2 = (float4)0;
    float4 b_2 = (float4)0;
    float4 c_2 = (float4)0;
    uint2 d_2 = (uint2)0;
    float4 local_2 = (float4)0;

    a_2 = ZeroValuefloat4();
    b_2 = ZeroValuefloat4();
    c_2 = ZeroValuefloat4();
    d_2 = ZeroValueuint2();
    float4 _e4 = nagaTextureSampleBaseClampToEdge(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, samp, (0.0).xx);
    a_2 = _e4;
    float4 _e8 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (int(0)).xx);
    b_2 = _e8;
    float4 _e12 = nagaTextureLoadExternal(tex_plane0_, tex_plane1_, tex_plane2_, tex_params, (0u).xx);
    c_2 = _e12;
    d_2 = NagaExternalDimensions2D(tex_plane0_, tex_plane1_, tex_plane2_, tex_params);
    float4 _e16 = a_2;
    float4 _e17 = b_2;
    float4 _e19 = c_2;
    uint2 _e21 = d_2;
    local_2 = (((_e16 + _e17) + _e19) + float2(_e21).xyxy);
    return;
}
