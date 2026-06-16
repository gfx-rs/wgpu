#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct _4
{
    uint _m0;
};

struct _10
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) readonly buffer _7_11
{
    uint _m0;
    int _m1[];
} _11[10];

layout(set = 0, binding = 10, std140) uniform _16_15
{
    _4 _m0;
} _15;

layout(location = 0) flat in uint _20;
layout(location = 0) out uint _23;

void main()
{
    uint _31 = 0u;
    uint _37 = _10(_20)._m0;
    _31 += _11[0u]._m0;
    uint _52;
    if (_15._m0._m0 < 1u)
    {
        _52 = _11[_15._m0._m0]._m0;
    }
    else
    {
        _52 = 0u;
    }
    _31 += _52;
    uint _60;
    if (_37 < 1u)
    {
        _60 = _11[nonuniformEXT(_37)]._m0;
    }
    else
    {
        _60 = 0u;
    }
    _31 += _60;
    _31 += uint(_11[0u]._m1.length());
    _31 += uint(_11[_15._m0._m0]._m1.length());
    _31 += uint(_11[_37]._m1.length());
    _23 = _31;
}

