#version 310 es

precision highp float;

struct VertexOutput {
    vec2 uv;
    vec4 position;
};

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    _fs2p_location0 = vec4(0.0, 0.5, 0.0, 0.5);
    return;
}

