#version 310 es

precision highp float;
precision highp int;

uniform highp samplerCube _group_0_binding_0_fs;


void main_1() {
    ivec2 sizeCube = ivec2(0);
    float a = 1.0;
    sizeCube = ivec2(uvec2(textureSize(_group_0_binding_0_fs, 0).xy));
    return;
}

void main() {
    main_1();
    return;
}

