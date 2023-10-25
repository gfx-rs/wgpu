#version 300 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec2 uv;
    vec4 position;
};
const float c_scale = 1.2;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    _fs2p_location0 = vec4(0.0, 0.5, 0.0, 0.5);
    return;
}

