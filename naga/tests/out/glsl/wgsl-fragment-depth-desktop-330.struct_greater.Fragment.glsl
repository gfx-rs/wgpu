#version 330 core
struct StructDepthOutput {
    float depth;
};

void main() {
    StructDepthOutput _tmp_return = StructDepthOutput(0.5);
    gl_FragDepth = _tmp_return.depth;
    return;
}
