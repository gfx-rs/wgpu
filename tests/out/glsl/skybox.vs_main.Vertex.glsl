#version 320 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec4 position;
    vec3 uv;
};

layout(std140, binding = 0) uniform Data_block_0Vs {
    mat4x4 proj_inv;
    mat4x4 view;
} _group_0_binding_0;

layout(location = 0) smooth out vec3 _vs2fs_location0;

void main() {
    uint vertex_index = uint(gl_VertexID);
    int tmp1 = 0;
    int tmp2 = 0;
    tmp1 = (int(vertex_index) / 2);
    tmp2 = (int(vertex_index) & 1);
    int _e10 = tmp1;
    int _e16 = tmp2;
    vec4 pos = vec4(((float(_e10) * 4.0) - 1.0), ((float(_e16) * 4.0) - 1.0), 0.0, 1.0);
    vec4 _e27 = _group_0_binding_0.view[0];
    vec4 _e31 = _group_0_binding_0.view[1];
    vec4 _e35 = _group_0_binding_0.view[2];
    mat3x3 inv_model_view = transpose(mat3x3(_e27.xyz, _e31.xyz, _e35.xyz));
    mat4x4 _e40 = _group_0_binding_0.proj_inv;
    vec4 unprojected = (_e40 * pos);
    VertexOutput _tmp_return = VertexOutput(pos, (inv_model_view * unprojected.xyz));
    gl_Position = _tmp_return.position;
    _vs2fs_location0 = _tmp_return.uv;
    return;
}

