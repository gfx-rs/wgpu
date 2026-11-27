#version 450
layout(location=0) flat in mat3x2 transforms[2];
layout(location=6) in Block {
    mat2 basis;
    vec2 offset;
} data;
layout(location=0) out vec4 color;
void main() {
    color = vec4(transforms[0][1] + transforms[1][2], data.basis[1] + data.offset);
}
