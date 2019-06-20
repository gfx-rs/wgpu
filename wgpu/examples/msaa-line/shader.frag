#version 450

layout(location = 0) in vec4 v_Color;
layout(location = 0) out vec4 f_Color;

void main() {
    f_Color = v_Color;
}
