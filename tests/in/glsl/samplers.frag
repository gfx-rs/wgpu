#version 440 core
precision mediump float;

layout(set = 1, binding = 0) uniform texture2DArray tex;
layout(set = 1, binding = 1) uniform sampler samp;

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_color;

void main() {
    o_color.rgba = texture(sampler2DArray(tex, samp), vec3(v_TexCoord, 0.0));
}
