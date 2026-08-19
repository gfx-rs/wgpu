#version 420 core
layout(depth_greater) out float gl_FragDepth;
struct StructDepthOutput {
    float depth;
};

void main() {
    gl_FragDepth = 0.5;
    return;
}
