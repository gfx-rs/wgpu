////////////////////////////////
// Entry point: "main" (vert) //
////////////////////////////////
#version 460

out float gl_ClipDistance[1];

struct _8
{
    vec4 _m0;
    float _m1[1];
};

struct _11
{
    vec4 _m0;
    float _m1[6];
    vec4 _m2;
};

void main()
{
    _8 _21 = _8(vec4(0.0), float[](0.0));
    _21._m1[0u] = 0.5;
    gl_Position = _21._m0;
    gl_ClipDistance = _21._m1;
}


//////////////////////////////////////
// Entry point: "main_split" (vert) //
//////////////////////////////////////
#version 460

out float gl_ClipDistance[6];

struct _8
{
    vec4 _m0;
    float _m1[1];
};

struct _11
{
    vec4 _m0;
    float _m1[6];
    vec4 _m2;
};

layout(location = 0) out vec4 _33;

void main()
{
    _11 _42 = _11(vec4(0.0), float[](0.0, 0.0, 0.0, 0.0, 0.0, 0.0), vec4(0.0));
    int _45 = 0;
    uvec2 _59 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _59)))
        {
            break;
        }
        _59 -= uvec2(uint(_59.y == 0u), 1u);
        if (!(_45 < 6))
        {
            break;
        }
        _42._m1[_45] = float(_45) - 2.5;
        _45++;
        continue;
    }
    _33 = _42._m0;
    gl_ClipDistance = _42._m1;
    gl_Position = _42._m2;
}

