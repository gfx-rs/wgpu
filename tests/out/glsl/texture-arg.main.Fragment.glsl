#version 310 es

precision highp float;

uniform highp sampler2D _group_0_binding_0;

layout(location = 0) out vec4 _fs2p_location0;

vec4 test(highp sampler2D Passed_Texture) {
    vec4 _expr7 = texture(Passed_Texture, vec2(vec2(0.0, 0.0)));
    return _expr7;
}

void main() {
    vec4 _expr2 = test(_group_0_binding_0);
    _fs2p_location0 = _expr2;
    return;
}

