#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

int _4(int _6, int _7)
{
    int _19 = ((_7 == 0) || ((_6 == int(0x80000000)) && (_7 == (-1)))) ? 1 : _7;
    return _6 - ((_6 / _19) * _19);
}

void main()
{
    int _32 = 5 / 2;
    uint _34 = 5u / 2u;
    uint _35 = 5u % 2u;
}

