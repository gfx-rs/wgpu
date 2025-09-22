#version 310 es
#extension GL_EXT_multiview : require

precision highp float;
precision highp int;


void main() {
    uint view_index = uint(gl_ViewIndex);
    return;
}

