#version 460

layout(set = 0, binding = 0) uniform texture2D Texture;
layout(set = 0, binding = 1) uniform sampler Sampler;

layout(location = 0) out vec4 _27;

vec4 test(texture2D Passed_Texture, sampler Passed_Sampler)
{
    return texture(sampler2D(Passed_Texture, Passed_Sampler), vec2(0.0));
}

void main()
{
    _27 = test(Texture, Sampler);
}

