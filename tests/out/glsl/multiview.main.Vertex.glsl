#version 310 es
#extension GL_EXT_multiview : require

precision highp float;
precision highp int;


void main() {
    int view_index = gl_ViewIndex;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

