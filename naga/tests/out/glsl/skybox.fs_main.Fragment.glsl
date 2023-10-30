#version 320 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec4 position;
    vec3 uv;
};
struct Data {
    mat4x4 proj_inv;
    mat4x4 view;
};
layout(binding = 0) uniform highp samplerCube _group_0_binding_1_fs;

layout(location = 0) smooth in vec3 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    VertexOutput in_ = VertexOutput(gl_FragCoord, _vs2fs_location0);
    vec4 _e4 = texture(_group_0_binding_1_fs, vec3(in_.uv));
    _fs2p_location0 = _e4;
    return;
}

