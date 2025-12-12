#version 320 es

precision highp float;
precision highp int;

struct ImmediateData {
    float multiplier;
};
struct FragmentIn {
    vec4 color;
};
uniform ImmediateData _immediates_binding_fs;

layout(location = 0) smooth in vec4 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    FragmentIn in_ = FragmentIn(_vs2fs_location0);
    float _e4 = _immediates_binding_fs.multiplier;
    _fs2p_location0 = (in_.color * _e4);
    return;
}

