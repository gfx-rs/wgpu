#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) nonuniformEXT flat in int v_Index;  // dynamically non-uniform
layout(location = 0) out vec4 o_Color;

layout(set = 0, binding = 0) uniform texture2D u_Textures[];
layout(set = 0, binding = 1) uniform sampler u_Sampler;

void main() {
    o_Color = vec4(texture(sampler2D(u_Textures[v_Index], u_Sampler), v_TexCoord).rgb, 1.0);
}
