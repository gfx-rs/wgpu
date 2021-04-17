#version 310 es

precision highp float;

struct VertexOutput {
    vec4 position;
    vec3 uv;
};

uniform highp samplerCube _group_0_binding_1;

smooth layout(location = 0) in vec3 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    VertexOutput in1 = VertexOutput(gl_FragCoord, _vs2fs_location0);
    vec4 _expr5 = texture(_group_0_binding_1, vec3(in1.uv));
    _fs2p_location0 = _expr5;
    return;
}

