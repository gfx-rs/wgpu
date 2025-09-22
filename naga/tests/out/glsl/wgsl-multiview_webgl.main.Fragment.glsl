#version 300 es
#extension GL_OVR_multiview2 : require

precision highp float;
precision highp int;


void main() {
    uint view_index = gl_ViewID_OVR;
    return;
}

