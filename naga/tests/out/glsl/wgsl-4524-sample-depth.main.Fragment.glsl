#version 310 es

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_fs;


void main() {
    float a = texture(_group_0_binding_0_fs, vec2(vec2(0.0))).r;
    return;
}

