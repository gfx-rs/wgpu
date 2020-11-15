#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    float x = float(gl_VertexIndex - 1);
    float y = float(((gl_VertexIndex & 1) * 2) - 1);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
