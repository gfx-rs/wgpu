#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

struct _4
{
    uint _m0;
};

struct _21
{
    uint _m0;
};

layout(set = 0, binding = 8, std140) uniform _43_42
{
    _4 _m0;
} _42;

layout(set = 0, binding = 0) uniform texture2D _24[10];
layout(set = 0, binding = 1) uniform texture2D _28[5];
layout(set = 0, binding = 2) uniform texture2DArray _30[5];
layout(set = 0, binding = 3) uniform texture2DMS _32[5];
layout(set = 0, binding = 4) uniform texture2D _34[5];
layout(set = 0, binding = 5, rgba32f) uniform writeonly image2D _36[5];
layout(set = 0, binding = 6) uniform sampler _38[5];
layout(set = 0, binding = 7) uniform samplerShadow _40[5];

layout(location = 0) flat in uint _47;
layout(location = 0) out vec4 _50;

void main()
{
    uvec2 _68 = uvec2(0u);
    vec4 _72 = vec4(0.0);
    uint _66 = 0u;
    float _70 = 0.0;
    uint _78 = _21(_47)._m0;
    _68 += uvec2(textureSize(_24[0u], int(0u)));
    _68 += uvec2(textureSize(_24[_42._m0._m0], int(0u)));
    _68 += uvec2(textureSize(_24[nonuniformEXT(_78)], int(0u)));
    _72 += textureGather(sampler2D(_28[0u], _38[0u]), vec2(0.0));
    _72 += textureGather(sampler2D(_28[_42._m0._m0], _38[_42._m0._m0]), vec2(0.0));
    _72 += textureGather(sampler2D(_28[_78], _38[_78]), vec2(0.0));
    _72 += textureGather(sampler2DShadow(_34[0u], _40[0u]), vec2(0.0), 0.0);
    _72 += textureGather(sampler2DShadow(_34[_42._m0._m0], _40[_42._m0._m0]), vec2(0.0), 0.0);
    _72 += textureGather(sampler2DShadow(_34[_78], _40[_78]), vec2(0.0), 0.0);
    vec4 _162;
    if (uint(0) < uint(textureQueryLevels(_24[0u])))
    {
        if (all(lessThan(uvec2(ivec2(0)), uvec2(textureSize(_24[0u], 0)))))
        {
            _162 = texelFetch(_24[0u], ivec2(0), 0);
        }
        else
        {
            _162 = vec4(0.0);
        }
    }
    else
    {
        _162 = vec4(0.0);
    }
    _72 += _162;
    vec4 _176;
    if (uint(0) < uint(textureQueryLevels(_24[_42._m0._m0])))
    {
        if (all(lessThan(uvec2(ivec2(0)), uvec2(textureSize(_24[_42._m0._m0], 0)))))
        {
            _176 = texelFetch(_24[_42._m0._m0], ivec2(0), 0);
        }
        else
        {
            _176 = vec4(0.0);
        }
    }
    else
    {
        _176 = vec4(0.0);
    }
    _72 += _176;
    vec4 _190;
    if (uint(0) < uint(textureQueryLevels(_24[nonuniformEXT(_78)])))
    {
        if (all(lessThan(uvec2(ivec2(0)), uvec2(textureSize(_24[nonuniformEXT(_78)], 0)))))
        {
            _190 = texelFetch(_24[nonuniformEXT(_78)], ivec2(0), 0);
        }
        else
        {
            _190 = vec4(0.0);
        }
    }
    else
    {
        _190 = vec4(0.0);
    }
    _72 += _190;
    _66 += uvec3(textureSize(_30[0u], int(0u))).z;
    _66 += uvec3(textureSize(_30[_42._m0._m0], int(0u))).z;
    _66 += uvec3(textureSize(_30[nonuniformEXT(_78)], int(0u))).z;
    _66 += uint(textureQueryLevels(_28[0u]));
    _66 += uint(textureQueryLevels(_28[_42._m0._m0]));
    _66 += uint(textureQueryLevels(_28[nonuniformEXT(_78)]));
    _66 += uint(textureSamples(_32[0u]));
    _66 += uint(textureSamples(_32[_42._m0._m0]));
    _66 += uint(textureSamples(_32[nonuniformEXT(_78)]));
    _72 += texture(sampler2D(_28[0u], _38[0u]), vec2(0.0));
    _72 += texture(sampler2D(_28[_42._m0._m0], _38[_42._m0._m0]), vec2(0.0));
    _72 += texture(sampler2D(_28[_78], _38[_78]), vec2(0.0));
    _72 += texture(sampler2D(_28[0u], _38[0u]), vec2(0.0), 0.0);
    _72 += texture(sampler2D(_28[_42._m0._m0], _38[_42._m0._m0]), vec2(0.0), 0.0);
    _72 += texture(sampler2D(_28[_78], _38[_78]), vec2(0.0), 0.0);
    _70 += texture(sampler2DShadow(_34[0u], _40[0u]), vec3(vec2(0.0), 0.0));
    _70 += texture(sampler2DShadow(_34[_42._m0._m0], _40[_42._m0._m0]), vec3(vec2(0.0), 0.0));
    _70 += texture(sampler2DShadow(_34[_78], _40[_78]), vec3(vec2(0.0), 0.0));
    _70 += textureLod(sampler2DShadow(_34[0u], _40[0u]), vec3(vec2(0.0), 0.0), 0.0);
    _70 += textureLod(sampler2DShadow(_34[_42._m0._m0], _40[_42._m0._m0]), vec3(vec2(0.0), 0.0), 0.0);
    _70 += textureLod(sampler2DShadow(_34[_78], _40[_78]), vec3(vec2(0.0), 0.0), 0.0);
    _72 += textureGrad(sampler2D(_28[0u], _38[0u]), vec2(0.0), vec2(0.0), vec2(0.0));
    _72 += textureGrad(sampler2D(_28[_42._m0._m0], _38[_42._m0._m0]), vec2(0.0), vec2(0.0), vec2(0.0));
    _72 += textureGrad(sampler2D(_28[_78], _38[_78]), vec2(0.0), vec2(0.0), vec2(0.0));
    _72 += textureLod(sampler2D(_28[0u], _38[0u]), vec2(0.0), 0.0);
    _72 += textureLod(sampler2D(_28[_42._m0._m0], _38[_42._m0._m0]), vec2(0.0), 0.0);
    _72 += textureLod(sampler2D(_28[_78], _38[_78]), vec2(0.0), 0.0);
    imageStore(_36[0u], ivec2(0), _72);
    imageStore(_36[_42._m0._m0], ivec2(0), _72);
    imageStore(_36[nonuniformEXT(_78)], ivec2(0), _72);
    _50 = (_72 + vec4(vec2(_68 + uvec2(_66)).xyxy)) + vec4(_70);
}

