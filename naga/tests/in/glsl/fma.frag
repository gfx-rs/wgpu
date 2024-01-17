#version 440 core

struct Mat4x3 { vec4 mx; vec4 my; vec4 mz; };
void Fma(inout Mat4x3 d, Mat4x3 m, float s) { d.mx += m.mx * s; d.my += m.my * s; d.mz += m.mz * s; }

out vec4 o_color;
void main() {
    o_color.rgba = vec4(1.0);
}
