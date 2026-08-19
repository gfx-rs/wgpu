#version 330 core
struct StructDepthOutput {
    float depth;
};

void main() {
    gl_FragDepth = 0.5;
    return;
}
