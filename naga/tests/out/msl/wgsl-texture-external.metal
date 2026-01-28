// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct NagaExternalTextureTransferFn {
    float a;
    float b;
    float g;
    float k;
};
struct NagaExternalTextureParams {
    metal::float4x4 yuv_conversion_matrix;
    metal::float3x3 gamut_conversion_matrix;
    NagaExternalTextureTransferFn src_tf;
    NagaExternalTextureTransferFn dst_tf;
    metal::float3x2 sample_transform;
    metal::float3x2 load_transform;
    metal::uint2 size;
    uint num_planes;
    char _pad8[4];
};
struct NagaExternalTextureWrapper {
    metal::texture2d<float, metal::access::sample> plane0;
    metal::texture2d<float, metal::access::sample> plane1;
    metal::texture2d<float, metal::access::sample> plane2;
    NagaExternalTextureParams params;
};

float4 nagaTextureSampleBaseClampToEdge(NagaExternalTextureWrapper tex, metal::sampler samp, float2 coords) {
    uint2 plane0_size = uint2(tex.plane0.get_width(), tex.plane0.get_height());
    coords = tex.params.sample_transform * float3(coords, 1.0);
    float2 bounds_min = tex.params.sample_transform * float3(0.0, 0.0, 1.0);
    float2 bounds_max = tex.params.sample_transform * float3(1.0, 1.0, 1.0);
    float4 bounds = float4(metal::min(bounds_min, bounds_max), metal::max(bounds_min, bounds_max));
    float2 plane0_half_texel = float2(0.5, 0.5) / float2(plane0_size);
    float2 plane0_coords = metal::clamp(coords, bounds.xy + plane0_half_texel, bounds.zw - plane0_half_texel);
    if (tex.params.num_planes == 1u) {
        return tex.plane0.sample(samp, plane0_coords, metal::level(0.0f));
    } else {
        uint2 plane1_size = uint2(tex.plane1.get_width(), tex.plane1.get_height());
        float2 plane1_half_texel = float2(0.5, 0.5) / float2(plane1_size);
        float2 plane1_coords = metal::clamp(coords, bounds.xy + plane1_half_texel, bounds.zw - plane1_half_texel);
        float y = tex.plane0.sample(samp, plane0_coords, metal::level(0.0f)).r;
        float2 uv = float2(0.0, 0.0);
        if (tex.params.num_planes == 2u) {
            uv = tex.plane1.sample(samp, plane1_coords, metal::level(0.0f)).xy;
        } else {
            uint2 plane2_size = uint2(tex.plane2.get_width(), tex.plane2.get_height());
            float2 plane2_half_texel = float2(0.5, 0.5) / float2(plane2_size);
            float2 plane2_coords = metal::clamp(coords, bounds.xy + plane2_half_texel, bounds.zw - plane1_half_texel);
            uv.x = tex.plane1.sample(samp, plane1_coords, metal::level(0.0f)).x;
            uv.y = tex.plane2.sample(samp, plane2_coords, metal::level(0.0f)).x;
        }
        float3 srcGammaRgb = (tex.params.yuv_conversion_matrix * float4(y, uv, 1.0)).rgb;
        float3 srcLinearRgb = metal::select(
            metal::pow((srcGammaRgb + tex.params.src_tf.a - 1.0) / tex.params.src_tf.a, tex.params.src_tf.g),
            srcGammaRgb / tex.params.src_tf.k,
            srcGammaRgb < tex.params.src_tf.k * tex.params.src_tf.b);
        float3 dstLinearRgb = tex.params.gamut_conversion_matrix * srcLinearRgb;
        float3 dstGammaRgb = metal::select(
            tex.params.dst_tf.a * metal::pow(dstLinearRgb, 1.0 / tex.params.dst_tf.g) - (tex.params.dst_tf.a - 1),
            tex.params.dst_tf.k * dstLinearRgb,
            dstLinearRgb < tex.params.dst_tf.b);
        return float4(dstGammaRgb, 1.0);
    }
}

float4 nagaTextureLoadExternal(NagaExternalTextureWrapper tex, uint2 coords) {
    uint2 plane0_size = uint2(tex.plane0.get_width(), tex.plane0.get_height());
    uint2 cropped_size = metal::any(tex.params.size != 0) ? tex.params.size : plane0_size;
    coords = metal::min(coords, cropped_size - 1);
    uint2 plane0_coords = uint2(metal::round(tex.params.load_transform * float3(float2(coords), 1.0)));
    if (tex.params.num_planes == 1u) {
        return tex.plane0.read(plane0_coords);
    } else {
        uint2 plane1_size = uint2(tex.plane1.get_width(), tex.plane1.get_height());
        uint2 plane1_coords = uint2(metal::floor(float2(plane0_coords) * float2(plane1_size) / float2(plane0_size)));
        float y = tex.plane0.read(plane0_coords).x;
        float2 uv;
        if (tex.params.num_planes == 2u) {
            uv = tex.plane1.read(plane1_coords).xy;
        } else {
        uint2 plane2_size = uint2(tex.plane2.get_width(), tex.plane2.get_height());
        uint2 plane2_coords = uint2(metal::floor(float2(plane0_coords) * float2(plane2_size) / float2(plane0_size)));
            uv = float2(tex.plane1.read(plane1_coords).x, tex.plane2.read(plane2_coords).x);
        }
        float3 srcGammaRgb = (tex.params.yuv_conversion_matrix * float4(y, uv, 1.0)).rgb;
        float3 srcLinearRgb = metal::select(
            metal::pow((srcGammaRgb + tex.params.src_tf.a - 1.0) / tex.params.src_tf.a, tex.params.src_tf.g),
            srcGammaRgb / tex.params.src_tf.k,
            srcGammaRgb < tex.params.src_tf.k * tex.params.src_tf.b);
        float3 dstLinearRgb = tex.params.gamut_conversion_matrix * srcLinearRgb;
        float3 dstGammaRgb = metal::select(
            tex.params.dst_tf.a * metal::pow(dstLinearRgb, 1.0 / tex.params.dst_tf.g) - (tex.params.dst_tf.a - 1),
            tex.params.dst_tf.k * dstLinearRgb,
            dstLinearRgb < tex.params.dst_tf.b);
        return float4(dstGammaRgb, 1.0);
    }
}

uint2 nagaTextureDimensionsExternal(NagaExternalTextureWrapper tex) {
    if (metal::any(tex.params.size != uint2(0u))) {
        return tex.params.size;
    } else {
        return uint2(tex.plane0.get_width(), tex.plane0.get_height());
    }
}

metal::float4 test(
    NagaExternalTextureWrapper t,
    metal::sampler samp
) {
    metal::float4 a = {};
    metal::float4 b = {};
    metal::float4 c = {};
    metal::uint2 d = {};
    metal::float4 _e4 = nagaTextureSampleBaseClampToEdge(t, samp, metal::float2(0.0));
    a = _e4;
    metal::float4 _e8 = nagaTextureLoadExternal(t, metal::uint2(metal::int2(0)));
    b = _e8;
    metal::float4 _e12 = nagaTextureLoadExternal(t, metal::uint2(metal::uint2(0u)));
    c = _e12;
    d = nagaTextureDimensionsExternal(t);
    metal::float4 _e16 = a;
    metal::float4 _e17 = b;
    metal::float4 _e19 = c;
    metal::uint2 _e21 = d;
    return ((_e16 + _e17) + _e19) + static_cast<metal::float2>(_e21).xyxy;
}

struct fragment_mainOutput {
    metal::float4 member [[color(0)]];
};
fragment fragment_mainOutput fragment_main(
  metal::texture2d<float, metal::access::sample> tex_plane0_ [[texture(0)]]
, metal::texture2d<float, metal::access::sample> tex_plane1_ [[texture(1)]]
, metal::texture2d<float, metal::access::sample> tex_plane2_ [[texture(2)]]
, constant NagaExternalTextureParams& tex_params [[buffer(0)]]
, metal::sampler samp [[sampler(0)]]
) {
    const NagaExternalTextureWrapper tex {
        .plane0 = tex_plane0_,
        .plane1 = tex_plane1_,
        .plane2 = tex_plane2_,
        .params = tex_params,
    };
    metal::float4 _e1 = test(tex, samp);
    return fragment_mainOutput { _e1 };
}


struct vertex_mainOutput {
    metal::float4 member_1 [[position]];
};
vertex vertex_mainOutput vertex_main(
  metal::texture2d<float, metal::access::sample> tex_plane0_ [[texture(0)]]
, metal::texture2d<float, metal::access::sample> tex_plane1_ [[texture(1)]]
, metal::texture2d<float, metal::access::sample> tex_plane2_ [[texture(2)]]
, constant NagaExternalTextureParams& tex_params [[buffer(0)]]
, metal::sampler samp [[sampler(0)]]
) {
    const NagaExternalTextureWrapper tex {
        .plane0 = tex_plane0_,
        .plane1 = tex_plane1_,
        .plane2 = tex_plane2_,
        .params = tex_params,
    };
    metal::float4 _e1 = test(tex, samp);
    return vertex_mainOutput { _e1 };
}


kernel void compute_main(
  metal::texture2d<float, metal::access::sample> tex_plane0_ [[texture(0)]]
, metal::texture2d<float, metal::access::sample> tex_plane1_ [[texture(1)]]
, metal::texture2d<float, metal::access::sample> tex_plane2_ [[texture(2)]]
, constant NagaExternalTextureParams& tex_params [[buffer(0)]]
, metal::sampler samp [[sampler(0)]]
) {
    const NagaExternalTextureWrapper tex {
        .plane0 = tex_plane0_,
        .plane1 = tex_plane1_,
        .plane2 = tex_plane2_,
        .params = tex_params,
    };
    metal::float4 _e1 = test(tex, samp);
    return;
}
