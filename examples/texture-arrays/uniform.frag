#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) flat in int v_Index;  // dynamically non-uniform
layout(location = 0) out vec4 o_Color;

layout(set = 0, binding = 0) uniform texture2D u_Textures[2];
layout(set = 0, binding = 1) uniform sampler u_Sampler;
layout(push_constant) uniform Uniforms {
    int u_Index;  // dynamically uniform
};

void main() {
    o_Color = vec4(texture(sampler2D(u_Textures[u_Index], u_Sampler), v_TexCoord).rgb, 1.0);
}
