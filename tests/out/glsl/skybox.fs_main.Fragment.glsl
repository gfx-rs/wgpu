#version 320 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec4 position;
    vec3 uv;
};

layout(binding = 0) uniform highp samplerCube _group_0_binding_1;

layout(location = 0) smooth in vec3 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    VertexOutput in1 = VertexOutput(gl_FragCoord, _vs2fs_location0);
    vec4 _e5 = texture(_group_0_binding_1, vec3(in1.uv));
    _fs2p_location0 = _e5;
    return;
}

