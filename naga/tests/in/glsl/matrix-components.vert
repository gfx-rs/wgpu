#version 450
void main() {
    mat3 m = mat3(1.0);
    m[0].y = 2.0;
    m[1].xz = vec2(3.0, 4.0);
    m[2].zyx = m[1].xyz;
    gl_Position = vec4(m[0].x, m[1].y, m[2].z, 1.0);
}
