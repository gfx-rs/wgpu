#version 300 es

precision highp float;
precision highp int;

struct VertexOutput {
    vec2 uv;
    vec4 position;
};

layout(location = 0) in vec2 _p2vs_location0;
layout(location = 1) in vec2 _p2vs_location1;
smooth out vec2 _vs2fs_location0;

void main() {
    vec2 pos = _p2vs_location0;
    vec2 uv = _p2vs_location1;
    VertexOutput _tmp_return = VertexOutput(uv, vec4((1.2 * pos), 0.0, 1.0));
    _vs2fs_location0 = _tmp_return.uv;
    gl_Position = _tmp_return.position;
    return;
}

