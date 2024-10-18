#version 450

void main() {
    vec4 a4 = vec4(1.0);
    vec4 b4 = vec4(2.0);
    mat4 m4 = mat4(a4, b4, a4, b4);

    vec3 a3 = vec3(1.0);
    vec3 b3 = vec3(2.0);
    mat3 m3 = mat3(a3, b3, a3);

    mat2 m2 = mat2(1.0, 2.0, 3.0, 4.0);

    mat4 m4_inverse = inverse(m4);
    mat3 m3_inverse = inverse(m3);
    mat2 m2_inverse = inverse(m2);
}