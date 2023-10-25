#version 320 es

precision highp float;
precision highp int;

struct PushConstants {
    float multiplier;
};
struct FragmentIn {
    vec4 color;
};
uniform PushConstants pc;

layout(location = 0) in vec2 _p2vs_location0;

void main() {
    vec2 pos = _p2vs_location0;
    uint vi = uint(gl_VertexID);
    float _e5 = pc.multiplier;
    gl_Position = vec4(((float(vi) * _e5) * pos), 0.0, 1.0);
    return;
}

