#version 450

layout(location = 0) in vec2 tex_coord;
layout(location = 1) flat in int index;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = vec4(tex_coord, 0.0, float(index));
}
