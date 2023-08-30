#version 310 es
#extension GL_EXT_blend_func_extended : require

precision highp float;
precision highp int;

struct FragmentOutput {
    vec4 color;
    vec4 mask;
};
layout(location = 0) out vec4 _fs2p_location0;
layout(location = 0, index = 1) out vec4 _fs2p_location1;

void main() {
    vec4 position = gl_FragCoord;
    vec4 color = vec4(0.0);
    vec4 mask = vec4(0.0);
    color = vec4(0.4, 0.3, 0.2, 0.1);
    mask = vec4(0.9, 0.8, 0.7, 0.6);
    vec4 _e13 = color;
    vec4 _e14 = mask;
    FragmentOutput _tmp_return = FragmentOutput(_e13, _e14);
    _fs2p_location0 = _tmp_return.color;
    _fs2p_location1 = _tmp_return.mask;
    return;
}

