#version 420 core
#extension GL_ARB_shader_draw_parameters : require
struct PushConstants {
    float multiplier;
};
struct FragmentIn {
    vec4 color;
};
uniform PushConstants _push_constant_binding_fs;

layout(location = 0) smooth in vec4 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    FragmentIn in_ = FragmentIn(_vs2fs_location0);
    float _e4 = _push_constant_binding_fs.multiplier;
    _fs2p_location0 = (in_.color * _e4);
    return;
}

