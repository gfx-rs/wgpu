#version 320 es

precision highp float;
precision highp int;

uniform uint naga_vs_first_instance;

struct PushConstants {
    float multiplier;
};
struct FragmentIn {
    vec4 color;
};
uniform PushConstants _push_constant_binding_vs;

layout(location = 0) in vec2 _p2vs_location0;

void main() {
    vec2 pos = _p2vs_location0;
    uint ii = (uint(gl_InstanceID) + naga_vs_first_instance);
    uint vi = uint(gl_VertexID);
    float _e8 = _push_constant_binding_vs.multiplier;
    gl_Position = vec4((((float(ii) * float(vi)) * _e8) * pos), 0.0, 1.0);
    return;
}

