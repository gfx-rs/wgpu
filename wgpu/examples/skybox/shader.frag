#version 450

layout(set = 0, binding = 1) uniform textureCube t_Cubemap;
layout(set = 0, binding = 2) uniform sampler s_Cubemap;

layout(location = 0) in vec3 v_Uv;
layout(location = 0) out vec4 f_Color;

void main() {
    f_Color = texture(samplerCube(t_Cubemap, s_Cubemap), v_Uv);
}
