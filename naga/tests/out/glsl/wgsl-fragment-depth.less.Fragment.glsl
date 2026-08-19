#version 420 core
layout(depth_less) out float gl_FragDepth;
struct StructDepthOutput {
    float depth;
};

void main() {
    gl_FragDepth = 0.5;
    return;
}
