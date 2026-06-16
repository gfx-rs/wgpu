/////////////////////////////////////////
// Entry point: "main_vec4vec3" (frag) //
/////////////////////////////////////////
#version 460

struct _12
{
    vec4 _m0;
    ivec4 _m1;
    uvec4 _m2;
    vec3 _m3;
    ivec3 _m4;
    uvec3 _m5;
};

struct _16
{
    vec2 _m0;
    ivec2 _m1;
    uvec2 _m2;
    float _m3;
    int _m4;
    uint _m5;
};

layout(location = 0) out vec4 _18;
layout(location = 1) out ivec4 _20;
layout(location = 2) out uvec4 _22;
layout(location = 3) out vec3 _24;
layout(location = 4) out ivec3 _26;
layout(location = 5) out uvec3 _28;

void main()
{
    _12 _41 = _12(vec4(0.0), ivec4(0), uvec4(0u), vec3(0.0), ivec3(0), uvec3(0u));
    _41._m0 = vec4(0.0);
    _41._m1 = ivec4(0);
    _41._m2 = uvec4(0u);
    _41._m3 = vec3(0.0);
    _41._m4 = ivec3(0);
    _41._m5 = uvec3(0u);
    _18 = _41._m0;
    _20 = _41._m1;
    _22 = _41._m2;
    _24 = _41._m3;
    _26 = _41._m4;
    _28 = _41._m5;
}


///////////////////////////////////////////
// Entry point: "main_vec2scalar" (frag) //
///////////////////////////////////////////
#version 460

struct _12
{
    vec4 _m0;
    ivec4 _m1;
    uvec4 _m2;
    vec3 _m3;
    ivec3 _m4;
    uvec3 _m5;
};

struct _16
{
    vec2 _m0;
    ivec2 _m1;
    uvec2 _m2;
    float _m3;
    int _m4;
    uint _m5;
};

layout(location = 0) out vec2 _70;
layout(location = 1) out ivec2 _72;
layout(location = 2) out uvec2 _74;
layout(location = 3) out float _76;
layout(location = 4) out int _78;
layout(location = 5) out uint _80;

void main()
{
    _16 _86 = _16(vec2(0.0), ivec2(0), uvec2(0u), 0.0, 0, 0u);
    _86._m0 = vec2(0.0);
    _86._m1 = ivec2(0);
    _86._m2 = uvec2(0u);
    _86._m3 = 0.0;
    _86._m4 = 0;
    _86._m5 = 0u;
    _70 = _86._m0;
    _72 = _86._m1;
    _74 = _86._m2;
    _76 = _86._m3;
    _78 = _86._m4;
    _80 = _86._m5;
}

