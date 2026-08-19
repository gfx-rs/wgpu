#version 420 core
layout(depth_greater) out float gl_FragDepth;
struct StructDepthOutput {
    float depth;
};

void main() {
    StructDepthOutput _tmp_return = StructDepthOutput(0.5);
    gl_FragDepth = _tmp_return.depth;
    return;
}
