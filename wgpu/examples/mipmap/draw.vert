#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 0) out vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_TexCoord = a_Pos.xy / 20.0 + 0.5;
    gl_Position = u_Transform * a_Pos;
}
