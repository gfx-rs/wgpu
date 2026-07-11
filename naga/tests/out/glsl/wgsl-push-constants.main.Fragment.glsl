#version 320 es

precision highp float;
precision highp int;

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
uniform ImmediateDataFrag _immediates_binding_fs;

layout(location = 0) smooth in vec4 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    FragmentIn in_ = FragmentIn(_vs2fs_location0);
    vec4 _e4 = _immediates_binding_fs.tint;
    _fs2p_location0 = (in_.color * _e4);
    return;
}

