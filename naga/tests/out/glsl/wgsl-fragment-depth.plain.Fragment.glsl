#version 420 core
struct StructDepthOutput {
    float depth;
};

void main() {
    gl_FragDepth = 0.5;
    return;
}
