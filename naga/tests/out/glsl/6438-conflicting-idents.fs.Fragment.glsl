#version 310 es

precision highp float;
precision highp int;

struct OurVertexShaderOutput {
    vec4 position;
    vec2 texcoord;
};
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    _fs2p_location0 = vec4(1.0, 0.0, 0.0, 1.0);
    return;
}

