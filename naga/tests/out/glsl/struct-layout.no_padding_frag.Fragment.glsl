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
layout(location = 0) smooth in vec3 _vs2fs_location0;
layout(location = 1) smooth in float _vs2fs_location1;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    NoPadding input_ = NoPadding(_vs2fs_location0, _vs2fs_location1);
    NoPadding _phony_0 = input_;
    _fs2p_location0 = vec4(0.0);
    return;
}

