#version 440 core
precision highp float;

layout(location = 0) out vec4 o_color;

float TevPerCompGT(float a, float b) {
    return float(a > b);
}

vec3 TevPerCompGT(vec3 a, vec3 b) {
    return vec3(greaterThan(a, b));
}

void main() {
    o_color.rgb = TevPerCompGT(vec3(3.0), vec3(5.0));
    o_color.a = TevPerCompGT(3.0, 5.0);
}
