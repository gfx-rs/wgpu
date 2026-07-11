#version 320 es

precision highp float;
precision highp int;

uniform uint naga_vs_first_instance;

struct ImmediateDataVert {
    float position_clip;
    mat3x3 matrix;
};
struct ImmediateDataFrag {
    float multiplier;
    vec4 tint;
};
struct FragmentIn {
    vec4 color;
};
uniform ImmediateDataVert _immediates_binding_vs;

layout(location = 0) in vec2 _p2vs_location0;

void main() {
    vec2 pos = _p2vs_location0;
    uint ii = (uint(gl_InstanceID) + naga_vs_first_instance);
    uint vi = uint(gl_VertexID);
    float _e9 = _immediates_binding_vs.position_clip;
    gl_Position = vec4(((float(ii) * float(vi)) * pos), 0.0, _e9);
    return;
}

