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
layout(location = 0) smooth in float _vs2fs_location0;
layout(location = 1) smooth in vec3 _vs2fs_location1;
layout(location = 2) smooth in float _vs2fs_location2;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    NeedsPadding input_2 = NeedsPadding(_vs2fs_location0, _vs2fs_location1, _vs2fs_location2);
    _fs2p_location0 = vec4(0.0);
    return;
}

