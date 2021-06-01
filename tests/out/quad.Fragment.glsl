#version 310 es

precision highp float;

struct VertexOutput {
    vec2 uv;
    vec4 position;
};

uniform highp sampler2D _group_0_binding_0;

smooth in vec2 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec2 uv2 = _vs2fs_location0;
    vec4 _expr4 = texture(_group_0_binding_0, vec2(uv2));
    if((_expr4.w == 0.0)) {
        discard;
    }
    _fs2p_location0 = (_expr4.w * _expr4);
    return;
}

