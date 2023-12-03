#version 310 es

precision highp float;
precision highp int;

vec4 color = vec4(0.0);

uniform highp sampler2D _group_0_binding_1_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main_1() {
    vec4 _e4 = texture(_group_0_binding_1_fs, vec2(vec2(0.0, 0.0)));
    color = _e4;
    return;
}

void main() {
    main_1();
    vec4 _e1 = color;
    _fs2p_location0 = _e1;
    return;
}

