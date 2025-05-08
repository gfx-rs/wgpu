#version 420 core
layout (depth_less) out float gl_FragDepth;

void main() {
    vec4 pos = gl_FragCoord;
    gl_FragDepth = (pos.z - 0.1);
    return;
}

