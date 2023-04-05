#version 310 es

precision highp float;
precision highp int;

struct gen_gl_PerVertex {
    vec4 gen_gl_Position;
    float gen_gl_PointSize;
    float gen_gl_ClipDistance[1];
    float gen_gl_CullDistance[1];
};
struct type_9 {
    vec2 member;
    vec4 gen_gl_Position;
};
vec2 v_uv = vec2(0.0);

vec2 a_uv_1 = vec2(0.0);

gen_gl_PerVertex perVertexStruct = gen_gl_PerVertex(vec4(0.0, 0.0, 0.0, 1.0), 1.0, float[1](0.0), float[1](0.0));

vec2 a_pos_1 = vec2(0.0);

layout(location = 1) in vec2 _p2vs_location1;
layout(location = 0) in vec2 _p2vs_location0;
layout(location = 0) smooth out vec2 _vs2fs_location0;

void main_1() {
    vec2 _e8 = a_uv_1;
    v_uv = _e8;
    vec2 _e9 = a_pos_1;
    perVertexStruct.gen_gl_Position = vec4(_e9.x, _e9.y, 0.0, 1.0);
    return;
}

void main() {
    vec2 a_uv = _p2vs_location1;
    vec2 a_pos = _p2vs_location0;
    a_uv_1 = a_uv;
    a_pos_1 = a_pos;
    main_1();
    vec2 _e7 = v_uv;
    vec4 _e8 = perVertexStruct.gen_gl_Position;
    type_9 _tmp_return = type_9(_e7, _e8);
    _vs2fs_location0 = _tmp_return.member;
    gl_Position = _tmp_return.gen_gl_Position;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

