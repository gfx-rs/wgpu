#version 310 es

precision highp float;
precision highp int;

struct type10 {
    vec2 member;
    vec4 gen_gl_Position;
    float gen_gl_PointSize;
    float gen_gl_ClipDistance[1];
    float gen_gl_CullDistance[1];
};

vec2 v_uv = vec2(0, 0);

vec2 a_uv1 = vec2(0, 0);

struct gen_gl_PerVertex_block_0 {
    vec4 gen_gl_Position;
    float gen_gl_PointSize;
    float gen_gl_ClipDistance[1];
    float gen_gl_CullDistance[1];
} perVertexStruct;

vec2 a_pos1 = vec2(0, 0);

layout(location = 1) in vec2 _p2vs_location1;
layout(location = 0) in vec2 _p2vs_location0;
layout(location = 0) smooth out vec2 _vs2fs_location0;

void main2() {
    vec2 _expr12 = a_uv1;
    v_uv = _expr12;
    vec2 _expr13 = a_pos1;
    perVertexStruct.gen_gl_Position = vec4(_expr13.x, _expr13.y, 0.0, 1.0);
    return;
}

void main() {
    vec2 a_uv = _p2vs_location1;
    vec2 a_pos = _p2vs_location0;
    a_uv1 = a_uv;
    a_pos1 = a_pos;
    main2();
    vec2 _expr10 = v_uv;
    vec4 _expr11 = perVertexStruct.gen_gl_Position;
    float _expr12 = perVertexStruct.gen_gl_PointSize;
    float _expr13[] = perVertexStruct.gen_gl_ClipDistance;
    float _expr14[] = perVertexStruct.gen_gl_CullDistance;
    type10 _tmp_return = type10(_expr10, _expr11, _expr12, _expr13, _expr14);
    _vs2fs_location0 = _tmp_return.member;
    gl_Position = _tmp_return.gen_gl_Position;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

