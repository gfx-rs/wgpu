#version 310 es
#extension GL_EXT_conservative_depth : require

precision highp float;
precision highp int;

layout (depth_less) out float gl_FragDepth;

void main() {
    vec4 pos = gl_FragCoord;
    gl_FragDepth = (pos.z - 0.1);
    return;
}

