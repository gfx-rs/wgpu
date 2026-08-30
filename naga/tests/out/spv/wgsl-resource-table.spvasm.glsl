#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 2, std140) uniform _18_17
{
    uint _m0;
} _17;

layout(set = 0, binding = 0) uniform sampler _14;
layout(set = 0, binding = 1) uniform samplerShadow _16;
layout(set = 1, binding = 0) uniform texture2D _22[];
layout(set = 1, binding = 0) uniform texture2D _25[];
layout(set = 1, binding = 0) uniform utexture2DArray _28[];
layout(set = 1, binding = 0) uniform texture2DMS _31[];

layout(location = 0) flat in uint _36;
layout(location = 0) out vec4 _39;

void main()
{
    _39 = ((((texture(sampler2D(_22[7u], _14), gl_FragCoord.xy) + texture(sampler2D(_22[_17._m0], _14), gl_FragCoord.xy)) + texture(sampler2D(_22[_36], _14), gl_FragCoord.xy)) + vec4(texture(sampler2DShadow(_25[_36 + 1u], _16), vec3(gl_FragCoord.xy, 0.5)))) + vec4(texelFetch(_28[nonuniformEXT(_17._m0)], ivec3(ivec2(0), 3), 0))) + texelFetch(_31[nonuniformEXT(0u)], ivec2(0), 2);
}

