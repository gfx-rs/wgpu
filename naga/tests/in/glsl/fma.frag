#version 440 core

struct Mat4x3 { vec4 mx; vec4 my; vec4 mz; };
void Fma(inout Mat4x3 d, Mat4x3 m, float s) { d.mx += m.mx * s; d.my += m.my * s; d.mz += m.mz * s; }

out vec4 o_color;
void main() {
    Mat4x3 m1 = {
        vec4(0),
        vec4(1),
        vec4(2),
    };
    Mat4x3 m2 = {
        vec4(0),
        vec4(1),
        vec4(2),
    };

    Fma(m1, m2, 2.0);
    o_color.rgba = vec4(1.0);
}
