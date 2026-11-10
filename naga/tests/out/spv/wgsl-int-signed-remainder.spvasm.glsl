#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

int _4(int _5, int _6)
{
    int _19 = ((_6 == 0) || ((_5 == int(0x80000000)) && (_6 == (-1)))) ? 1 : _6;
    return _5 - ((_5 / _19) * _19);
}

ivec2 _24(ivec2 _25, ivec2 _26)
{
    bvec2 _31 = equal(_26, ivec2(0));
    bvec2 _34 = equal(_25, ivec2(int(0x80000000)));
    bvec2 _35 = equal(_26, ivec2(-1));
    bvec2 _36 = bvec2(_34.x && _35.x, _34.y && _35.y);
    ivec2 _39 = mix(_26, ivec2(1), bvec2(_31.x || _36.x, _31.y || _36.y));
    return _25 - ((_25 / _39) * _39);
}

uint _44(uint _45, uint _46)
{
    return _45 % ((_46 == 0u) ? 1u : _46);
}

void main()
{
}

