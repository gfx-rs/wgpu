#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct _4
{
    uint _m0;
};

struct _5
{
    uint _m0;
};

struct _11
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) readonly buffer _8_12
{
    uint _m0;
    _5 _m1;
    int _m2[];
} _12[10];

layout(set = 0, binding = 10, std140) uniform _17_16
{
    _4 _m0;
} _16;

layout(location = 0) flat in uint _21;
layout(location = 0) out uint _24;

void main()
{
    uint _32 = 0u;
    uint _38 = _11(_21)._m0;
    _32 += _12[0u]._m0;
    uint _53;
    if (_16._m0._m0 < 1u)
    {
        _53 = _12[_16._m0._m0]._m0;
    }
    else
    {
        _53 = 0u;
    }
    _32 += _53;
    uint _61;
    if (_38 < 1u)
    {
        _61 = _12[nonuniformEXT(_38)]._m0;
    }
    else
    {
        _61 = 0u;
    }
    _32 += _61;
    _32 += _12[0u]._m1._m0;
    uint _74;
    if (_16._m0._m0 < 1u)
    {
        _74 = _12[_16._m0._m0]._m1._m0;
    }
    else
    {
        _74 = 0u;
    }
    _32 += _74;
    uint _82;
    if (_38 < 1u)
    {
        _82 = _12[nonuniformEXT(_38)]._m1._m0;
    }
    else
    {
        _82 = 0u;
    }
    _32 += _82;
    _32 += uint(_12[0u]._m2.length());
    _32 += uint(_12[_16._m0._m0]._m2.length());
    _32 += uint(_12[_38]._m2.length());
    _24 = _32;
}

