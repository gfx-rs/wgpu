#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer type_1_block_0Compute { float _group_0_binding_0_cs[4]; };

layout(std430) buffer type_3_block_1Compute { int _group_0_binding_1_cs[4]; };


void main() {
    float _e4 = _group_0_binding_0_cs[1];
    _group_0_binding_0_cs[0] = sign(_e4);
    float _e8 = _group_0_binding_0_cs[2];
    float _e11 = _group_0_binding_0_cs[3];
    vec2 v = vec2(_e8, _e11);
    vec2 signs = sign(v);
    _group_0_binding_0_cs[2] = signs.x;
    _group_0_binding_0_cs[3] = signs.y;
    int _e24 = _group_0_binding_1_cs[1];
    _group_0_binding_1_cs[0] = sign(_e24);
    return;
}

