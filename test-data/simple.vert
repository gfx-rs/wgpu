#version 450 core

layout(location = 0) in vec2 position;

void main() {
    float w = 1.0;
    gl_Position = vec4(position, 0.0, w);
}
