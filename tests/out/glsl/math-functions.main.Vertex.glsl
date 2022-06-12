#version 310 es

precision highp float;
precision highp int;


void main() {
    vec4 v = vec4(0.0);
    float a = degrees(1.0);
    float b = radians(1.0);
    vec4 c = degrees(v);
    vec4 d = radians(v);
    int const_dot = ( + ivec2(0, 0).x * ivec2(0, 0).x + ivec2(0, 0).y * ivec2(0, 0).y);
    uint first_leading_bit_abs = uint(findMSB(uint(abs(int(0u)))));
}

