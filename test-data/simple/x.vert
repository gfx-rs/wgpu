#version 450 core

layout(location = 0) in vec2 a_pos;
layout(location = 0) out vec4 o_pos;

void main() {
    float w = 1.0;
    o_pos = vec4(a_pos, 0.0, w);
    return;
}
