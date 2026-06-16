#version 460

layout(set = 0, binding = 0) uniform texture2D _8;
layout(set = 0, binding = 1) uniform sampler _10;

layout(location = 0) in vec2 _13;
layout(location = 0) out vec4 _16;

void main()
{
    _16 = texture(sampler2D(_8, _10), _13);
}

