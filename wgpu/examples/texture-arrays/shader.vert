#version 450

layout(location = 0) in vec2 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in int a_Index;
layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) flat out int v_Index;

void main() {
    v_TexCoord = a_TexCoord;
    v_Index = a_Index;
    gl_Position = vec4(a_Pos, 0.0, 1.0);
}
