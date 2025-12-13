#version 310 es

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_fs;

layout(location = 0) out vec4 _fs2p_location0;

vec4 test(highp sampler2D Passed_Texture) {
    vec4 _e5 = texture(Passed_Texture, vec2(vec2(0.0, 0.0)));
    return _e5;
}

void main() {
    vec4 _e2 = test(_group_0_binding_0_fs);
    _fs2p_location0 = _e2;
    return;
}

