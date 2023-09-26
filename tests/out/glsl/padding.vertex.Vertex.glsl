#version 310 es

precision highp float;
precision highp int;

struct S {
    vec3 a;
};
struct Test {
    S a;
    float b;
};
struct Test2_ {
    vec3 a[2];
    float b;
};
struct Test3_ {
    mat4x3 a;
    float b;
};
uniform Test_block_0Vertex { Test _group_0_binding_0_vs; };

uniform Test2_block_1Vertex { Test2_ _group_0_binding_1_vs; };

uniform Test3_block_2Vertex { Test3_ _group_0_binding_2_vs; };


void main() {
    float _e4 = _group_0_binding_0_vs.b;
    float _e8 = _group_0_binding_1_vs.b;
    float _e12 = _group_0_binding_2_vs.b;
    gl_Position = (((vec4(1.0) * _e4) * _e8) * _e12);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

