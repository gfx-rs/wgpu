///////////////////////////////////
// Entry point: "vs_main" (vert) //
///////////////////////////////////
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform texture2D _13[];

layout(location = 0) in vec2 _28;

void main()
{
    gl_Position = vec4(_28, 0.0, 1.0);
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform sampler _9;
layout(set = 1, binding = 0) uniform texture2D _13[];

layout(location = 0) flat in uint _43;
layout(location = 0) out vec4 _46;

vec4 _17(uint _15, vec2 _16)
{
    return texture(sampler2D(_13[_15], _9), _16);
}

void main()
{
    _46 = _17(_43, gl_FragCoord.xy);
}

