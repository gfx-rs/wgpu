#version 310 es

precision highp float;
precision highp int;

struct NoPadding {
    vec3 v3_;
    float f3_;
};
struct NeedsPadding {
    float f3_forces_padding;
    vec3 v3_needs_padding;
    float f3_;
};
layout(location = 0) in float _p2vs_location0;
layout(location = 1) in vec3 _p2vs_location1;
layout(location = 2) in float _p2vs_location2;

void main() {
    NeedsPadding input_3 = NeedsPadding(_p2vs_location0, _p2vs_location1, _p2vs_location2);
    gl_Position = vec4(0.0);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

