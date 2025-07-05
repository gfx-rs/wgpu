#version 450
#extension GL_EXT_mesh_shader : require

in VertexInput { layout(location = 0) vec4 color; }
vertexInput;

layout(location = 0) out vec4 fragColor;

void main() { fragColor = vertexInput.color; }