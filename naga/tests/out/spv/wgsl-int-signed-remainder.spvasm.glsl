#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

int _4(int _6, int _7)
{
    int _19 = ((_7 == 0) || ((_6 == int(0x80000000)) && (_7 == (-1)))) ? 1 : _7;
    return _6 - ((_6 / _19) * _19);
}

ivec2 _24(ivec2 _26, ivec2 _27)
{
    bvec2 _31 = equal(_27, ivec2(0));
    bvec2 _34 = equal(_26, ivec2(int(0x80000000)));
    bvec2 _35 = equal(_27, ivec2(-1));
    bvec2 _36 = bvec2(_34.x && _35.x, _34.y && _35.y);
    ivec2 _39 = mix(_27, ivec2(1), bvec2(_31.x || _36.x, _31.y || _36.y));
    return _26 - ((_26 / _39) * _39);
}

uint _44(uint _46, uint _47)
{
    return _46 % ((_47 == 0u) ? 1u : _47);
}

void main()
{
}

