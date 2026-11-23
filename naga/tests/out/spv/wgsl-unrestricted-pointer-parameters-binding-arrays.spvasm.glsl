#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D _13[4];
layout(set = 0, binding = 1) uniform sampler _15[4];

layout(location = 0) in vec2 _18;
layout(location = 0) out vec4 _21;

void main()
{
    uvec2 _59 = uvec2(0u);
    vec4 _53 = vec4(0.0);
    vec4 _47 = vec4(0.0);
    vec2 _41 = vec2(0.0);
    uint _32 = 0u;
    vec4 _55 = vec4(0.0);
    uint _49 = 0u;
    uint _43 = 0u;
    vec2 _35 = vec2(0.0);
    uint _57 = 0u;
    vec2 _51 = vec2(0.0);
    vec2 _45 = vec2(0.0);
    vec4 _38 = vec4(0.0);
    _32 = 0u;
    _35 = _18;
    _38 = texture(sampler2D(_13[_32], _15[_32]), _35);
    _41 = _18;
    _43 = 0u;
    _45 = vec2(0.0);
    _47 = vec4(0.0);
    _49 = 0u;
    _51 = vec2(0.0);
    _53 = vec4(0.0);
    _43 = 1u;
    _45 = _41;
    _47 = texture(sampler2D(_13[_43], _15[_43]), _45);
    _49 = 2u;
    _51 = _41;
    _53 = texture(sampler2D(_13[_49], _15[_49]), _51);
    _55 = _47 + _53;
    _57 = 3u;
    _59 = uvec2(textureSize(_13[nonuniformEXT(_57)], int(0u)));
    _21 = (_38 + _55) * float(_59.x);
}

