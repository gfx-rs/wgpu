#version 310 es

precision highp float;
precision highp int;

struct _modf_result_f32_ {
    float fract_;
    float whole;
};
struct _modf_result_vec2_f32_ {
    vec2 fract_;
    vec2 whole;
};
struct _modf_result_vec4_f32_ {
    vec4 fract_;
    vec4 whole;
};
struct _frexp_result_f32_ {
    float fract_;
    int exp_;
};
struct _frexp_result_vec4_f32_ {
    vec4 fract_;
    ivec4 exp_;
};

_modf_result_f32_ naga_modf(float arg) {
    float other;
    float fract = modf(arg, other);
    return _modf_result_f32_(fract, other);
}

_modf_result_vec2_f32_ naga_modf(vec2 arg) {
    vec2 other;
    vec2 fract = modf(arg, other);
    return _modf_result_vec2_f32_(fract, other);
}

_modf_result_vec4_f32_ naga_modf(vec4 arg) {
    vec4 other;
    vec4 fract = modf(arg, other);
    return _modf_result_vec4_f32_(fract, other);
}

_frexp_result_f32_ naga_frexp(float arg) {
    int other;
    float fract = frexp(arg, other);
    return _frexp_result_f32_(fract, other);
}

_frexp_result_vec4_f32_ naga_frexp(vec4 arg) {
    ivec4 other;
    vec4 fract = frexp(arg, other);
    return _frexp_result_vec4_f32_(fract, other);
}

void main() {
    vec4 v = vec4(0.0);
    float a = degrees(1.0);
    float b = radians(1.0);
    vec4 c = degrees(v);
    vec4 d = radians(v);
    vec4 e = clamp(v, vec4(0.0), vec4(1.0));
    vec4 g = refract(v, v, 1.0);
    ivec4 sign_b = ivec4(-1, -1, -1, -1);
    vec4 sign_d = vec4(-1.0, -1.0, -1.0, -1.0);
    int const_dot = ( + ivec2(0).x * ivec2(0).x + ivec2(0).y * ivec2(0).y);
    ivec2 flb_b = ivec2(-1, -1);
    uvec2 flb_c = uvec2(0u, 0u);
    ivec2 ftb_c = ivec2(0, 0);
    uvec2 ftb_d = uvec2(0u, 0u);
    uvec2 ctz_e = uvec2(32u, 32u);
    ivec2 ctz_f = ivec2(32, 32);
    uvec2 ctz_g = uvec2(0u, 0u);
    ivec2 ctz_h = ivec2(0, 0);
    ivec2 clz_c = ivec2(0, 0);
    uvec2 clz_d = uvec2(31u, 31u);
    float lde_a = ldexp(1.0, 2);
    vec2 lde_b = ldexp(vec2(1.0, 2.0), ivec2(3, 4));
    _modf_result_f32_ modf_a = naga_modf(1.5);
    float modf_b = naga_modf(1.5).fract_;
    float modf_c = naga_modf(1.5).whole;
    _modf_result_vec2_f32_ modf_d = naga_modf(vec2(1.5, 1.5));
    float modf_e = naga_modf(vec4(1.5, 1.5, 1.5, 1.5)).whole.x;
    float modf_f = naga_modf(vec2(1.5, 1.5)).fract_.y;
    _frexp_result_f32_ frexp_a = naga_frexp(1.5);
    float frexp_b = naga_frexp(1.5).fract_;
    int frexp_c = naga_frexp(1.5).exp_;
    int frexp_d = naga_frexp(vec4(1.5, 1.5, 1.5, 1.5)).exp_.x;
    float quantizeToF16_a = unpackHalf2x16(packHalf2x16(vec2(1.0))).x;
    vec2 _e120 = vec2(1.0, 1.0);
    vec2 quantizeToF16_b = unpackHalf2x16(packHalf2x16(_e120));
    vec3 _e125 = vec3(1.0, 1.0, 1.0);
    vec3 quantizeToF16_c = vec3(unpackHalf2x16(packHalf2x16(_e125.xy)), unpackHalf2x16(packHalf2x16(_e125.zz)).x);
    vec4 _e131 = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 quantizeToF16_d = vec4(unpackHalf2x16(packHalf2x16(_e131.xy)), unpackHalf2x16(packHalf2x16(_e131.zw)));
}

