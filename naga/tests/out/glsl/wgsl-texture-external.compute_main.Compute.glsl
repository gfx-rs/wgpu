#version 310 es
#extension GL_OES_EGL_image_external_essl3 : require

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct NagaExternalTextureTransferFn {
    float a;
    float b;
    float g;
    float k;
};
struct NagaExternalTextureParams {
    mat4x4 yuv_conversion_matrix;
    mat3x3 gamut_conversion_matrix;
    NagaExternalTextureTransferFn src_tf;
    NagaExternalTextureTransferFn dst_tf;
    mat3x2 sample_transform;
    mat3x2 load_transform;
    uvec2 size;
    uint num_planes;
};
layout(binding = 0) uniform highp samplerExternalOES _group_0_binding_0_cs;

layout(std430, binding = 0) readonly buffer NagaExternalTextureParams_block_0Compute { NagaExternalTextureParams _member; } _group_0_binding_0_cs_params;


vec4 nagaSampleExternalTexture(highp samplerExternalOES tex, NagaExternalTextureParams params, vec2 coords) {
    coords = (params.sample_transform * vec3(coords, 1.0));
    vec2 bounds_min = (params.sample_transform * vec3(0.0, 0.0, 1.0));
    vec2 bounds_max = (params.sample_transform * vec3(1.0, 1.0, 1.0));
    vec4 bounds = vec4(min(bounds_min, bounds_max), max(bounds_min, bounds_max));
    vec2 size = vec2(textureSize(tex, 0));
    vec2 half_texel = vec2(0.5) / size;
    vec2 clamped = clamp(coords, bounds.xy + half_texel, bounds.zw - half_texel);
    vec4 srcColor = texture(tex, clamped);
    vec3 srcGammaRgb = srcColor.rgb;
    vec3 srcLinearRgb = mix(
        pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, vec3(params.src_tf.g)),
        srcGammaRgb / params.src_tf.k,
        lessThan(srcGammaRgb, vec3(params.src_tf.k * params.src_tf.b)));
    vec3 dstLinearRgb = params.gamut_conversion_matrix * srcLinearRgb;
    vec3 dstGammaRgb = mix(
        params.dst_tf.a * pow(dstLinearRgb, vec3(1.0 / params.dst_tf.g)) - (params.dst_tf.a - 1.0),
        params.dst_tf.k * dstLinearRgb,
        lessThan(dstLinearRgb, vec3(params.dst_tf.b)));
    return vec4(dstGammaRgb, srcColor.a);
}

vec4 nagaTextureLoadExternal(highp samplerExternalOES tex, NagaExternalTextureParams params, ivec2 coords) {
    uvec2 tex_size = uvec2(textureSize(tex, 0));
    uvec2 cropped_size = (params.size != uvec2(0u)) ? params.size : tex_size;
    coords = min(coords, ivec2(cropped_size - uvec2(1u)));
    ivec2 transformed = ivec2(round(params.load_transform * vec3(vec2(coords), 1.0)));
    vec4 srcColor = texelFetch(tex, transformed, 0);
    vec3 srcGammaRgb = srcColor.rgb;
    vec3 srcLinearRgb = mix(
        pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, vec3(params.src_tf.g)),
        srcGammaRgb / params.src_tf.k,
        lessThan(srcGammaRgb, vec3(params.src_tf.k * params.src_tf.b)));
    vec3 dstLinearRgb = params.gamut_conversion_matrix * srcLinearRgb;
    vec3 dstGammaRgb = mix(
        params.dst_tf.a * pow(dstLinearRgb, vec3(1.0 / params.dst_tf.g)) - (params.dst_tf.a - 1.0),
        params.dst_tf.k * dstLinearRgb,
        lessThan(dstLinearRgb, vec3(params.dst_tf.b)));
    return vec4(dstGammaRgb, srcColor.a);
}

uvec2 nagaTextureDimensionsExternal(highp samplerExternalOES tex, NagaExternalTextureParams params) {
    uvec2 s = params.size;
    return (s != uvec2(0u)) ? s : uvec2(textureSize(tex, 0));
}

vec4 test(highp samplerExternalOES t, NagaExternalTextureParams t_params) {
    vec4 a = vec4(0.0);
    vec4 b = vec4(0.0);
    vec4 c = vec4(0.0);
    uvec2 d = uvec2(0u);
    vec4 _e4 = nagaSampleExternalTexture(t, t_params, vec2(0.0));
    a = _e4;
    vec4 _e8 = nagaTextureLoadExternal(t, t_params, ivec2(ivec2(0)));
    b = _e8;
    vec4 _e12 = nagaTextureLoadExternal(t, t_params, ivec2(uvec2(0u)));
    c = _e12;
    d = uvec2(nagaTextureDimensionsExternal(t, t_params).xy);
    vec4 _e16 = a;
    vec4 _e17 = b;
    vec4 _e19 = c;
    uvec2 _e21 = d;
    return (((_e16 + _e17) + _e19) + vec2(_e21).xyxy);
}

void main() {
    vec4 _e1 = test(_group_0_binding_0_cs, _group_0_binding_0_cs_params._member);
    return;
}

