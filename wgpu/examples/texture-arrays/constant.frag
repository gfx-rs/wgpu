#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) flat in int v_Index;  // dynamically non-uniform
layout(location = 0) out vec4 o_Color;

layout(set = 0, binding = 0) uniform texture2D u_Textures[2];
layout(set = 0, binding = 1) uniform sampler u_Sampler;

void main() {
    if (v_Index == 0) {
        o_Color = vec4(texture(sampler2D(u_Textures[0], u_Sampler), v_TexCoord).rgb, 1.0);
    } else if (v_Index == 1) {
        o_Color = vec4(texture(sampler2D(u_Textures[1], u_Sampler), v_TexCoord).rgb, 1.0);
    } else {
        // We need to write something to output color
        o_Color = vec4(0.0, 0.0, 1.0, 0.0);
    }
}
