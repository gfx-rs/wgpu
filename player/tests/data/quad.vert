#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    vec2 pos = vec2(gl_VertexIndex == 2 ? 3.0 : -1.0, gl_VertexIndex == 1 ? 3.0 : -1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}