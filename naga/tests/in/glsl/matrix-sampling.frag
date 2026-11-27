#version 450
layout(location=0) centroid in mat2 transforms[2];
layout(location=4) sample in mat2 basis;
layout(location=6) in Block {
    vec2 offset;
} data;
layout(location=7) centroid in vec2 bias;
layout(location=8) sample in float scale;
layout(location=0) out vec4 color;
void main() {
    color = vec4((transforms[0][0] + transforms[1][1] + basis[0] + data.offset + bias) * scale, 0.0, 1.0);
}
