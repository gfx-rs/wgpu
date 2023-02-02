#version 310 es

precision highp float;
precision highp int;


void main() {
    vec4 v = vec4(0.0);
    float a = degrees(1.0);
    float b = radians(1.0);
    vec4 c = degrees(v);
    vec4 d = radians(v);
    vec4 e = clamp(v, vec4(0.0), vec4(1.0));
    vec4 g = refract(v, v, 1.0);
    int const_dot = ( + ivec2(0, 0).x * ivec2(0, 0).x + ivec2(0, 0).y * ivec2(0, 0).y);
    uint first_leading_bit_abs = uint(findMSB(uint(abs(int(0u)))));
    int clz_a = (-1 < 0 ? 0 : 31 - findMSB(-1));
    uint clz_b = uint(31 - findMSB(1u));
    ivec2 _e20 = ivec2(-1);
    ivec2 clz_c = mix(ivec2(31) - findMSB(_e20), ivec2(0), lessThan(_e20, ivec2(0)));
    uvec2 clz_d = uvec2(ivec2(31) - findMSB(uvec2(1u)));
}

