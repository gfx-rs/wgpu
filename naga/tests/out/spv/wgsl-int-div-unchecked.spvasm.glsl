#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

int _4(int _5, int _6)
{
    int _19 = ((_6 == 0) || ((_5 == int(0x80000000)) && (_6 == (-1)))) ? 1 : _6;
    return _5 - ((_5 / _19) * _19);
}

void main()
{
    int _32 = 5 / 2;
    uint _34 = 5u / 2u;
    uint _35 = 5u % 2u;
}

