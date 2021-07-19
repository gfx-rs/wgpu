#version 320 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec4 position;
    vec3 uv;
};

layout(binding = 0) uniform Data_block_0 {
    mat4x4 proj_inv;
    mat4x4 view;
} _group_0_binding_0;

layout(location = 0) smooth out vec3 _vs2fs_location0;

void main() {
    uint vertex_index = uint(gl_VertexID);
    int tmp1_;
    int tmp2_;
    tmp1_ = (int(vertex_index) / 2);
    tmp2_ = (int(vertex_index) & 1);
    int _expr10 = tmp1_;
    int _expr16 = tmp2_;
    vec4 pos = vec4(((float(_expr10) * 4.0) - 1.0), ((float(_expr16) * 4.0) - 1.0), 0.0, 1.0);
    vec4 _expr27 = _group_0_binding_0.view[0];
    vec4 _expr31 = _group_0_binding_0.view[1];
    vec4 _expr35 = _group_0_binding_0.view[2];
    mat3x3 inv_model_view = transpose(mat3x3(_expr27.xyz, _expr31.xyz, _expr35.xyz));
    mat4x4 _expr40 = _group_0_binding_0.proj_inv;
    vec4 unprojected = (_expr40 * pos);
    VertexOutput _tmp_return = VertexOutput(pos, (inv_model_view * unprojected.xyz));
    gl_Position = _tmp_return.position;
    _vs2fs_location0 = _tmp_return.uv;
    return;
}

