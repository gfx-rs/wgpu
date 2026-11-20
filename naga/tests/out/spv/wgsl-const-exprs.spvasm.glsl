#version 460
layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;

const int _159[9] = int[](1, 2, 3, 4, 5, 6, 7, 8, 9);

void _40()
{
    ivec4 _43 = ivec4(4, 3, 2, 1);
}

void _47()
{
    int _48 = 2;
}

void _52()
{
    int _54 = 6;
}

void _57()
{
    ivec4 _66 = ivec4(0);
    int _61 = 0;
    int _65 = 70;
    int _60 = 30;
    int _63 = 0;
    _61 = _60;
    _63 = _61;
    _66 = ivec4(_60, _61, _63, _65);
}

void _77()
{
    ivec4 _80 = ivec4(-4);
}

void _83()
{
    ivec4 _84 = ivec4(-4);
}

uint _88(int _87)
{
    switch (_87)
    {
        case 0:
        {
            return 10u;
        }
        case 1:
        {
            return 20u;
        }
        case 2:
        {
            return 30u;
        }
        default:
        {
            return 0u;
        }
    }
    return 0u;
}

void _102()
{
    vec4 _105 = vec4(2.0, 1.0, 1.0, 1.0);
}

void _109()
{
    float _110[2] = float[](0.0, 0.0);
}

void _115()
{
    ivec3 _119 = ivec3(1);
    ivec3 _121 = ivec3(0, 1, 2);
    ivec3 _122 = ivec3(1, 0, 2);
}

void _125()
{
    bool _133 = false;
    bool _130 = true;
    bool _126 = false;
    bool _134 = true;
    bool _131 = false;
    bool _128 = true;
    bool _132 = true;
    bool _129 = false;
}

void _137()
{
    int _147 = 70;
    uint _143 = 4u;
    int _149 = -4;
    uint _146 = 12u;
    int _142 = 4;
    uint _148 = 70u;
    int _145 = 12;
}

void _153(uint _152)
{
    uint _163 = 1u;
    int _166 = 0;
    float _161 = 1.0;
    int _164 = 0;
    _164 = _159[_152];
    _166 = ivec4(1, 2, 3, 4)[_152];
}

void main()
{
    _40();
    _47();
    _52();
    _57();
    _77();
    _83();
    _102();
    _109();
    _115();
    _125();
    _137();
    _109();
    _153(1u);
}

