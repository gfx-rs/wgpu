#version 460
layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;

const int _154[9] = int[](1, 2, 3, 4, 5, 6, 7, 8, 9);

void _34()
{
    ivec4 _37 = ivec4(4, 3, 2, 1);
}

void _41()
{
    int _42 = 2;
}

void _46()
{
    int _48 = 6;
}

void _51()
{
    ivec4 _60 = ivec4(0);
    int _55 = 0;
    int _59 = 70;
    int _54 = 30;
    int _57 = 0;
    _55 = _54;
    _57 = _55;
    _60 = ivec4(_54, _55, _57, _59);
}

void _71()
{
    ivec4 _74 = ivec4(-4);
}

void _77()
{
    ivec4 _78 = ivec4(-4);
}

uint _82(int _81)
{
    switch (_81)
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

void _96()
{
    vec4 _100 = vec4(2.0, 1.0, 1.0, 1.0);
}

void _104()
{
    float _105[2] = float[](0.0, 0.0);
}

void _110()
{
    ivec3 _114 = ivec3(1);
    ivec3 _116 = ivec3(0, 1, 2);
    ivec3 _117 = ivec3(1, 0, 2);
}

void _120()
{
    bool _128 = false;
    bool _125 = true;
    bool _121 = false;
    bool _129 = true;
    bool _126 = false;
    bool _123 = true;
    bool _127 = true;
    bool _124 = false;
}

void _132()
{
    int _142 = 70;
    uint _138 = 4u;
    int _144 = -4;
    uint _141 = 12u;
    int _137 = 4;
    uint _143 = 70u;
    int _140 = 12;
}

void _148(uint _147)
{
    uint _158 = 1u;
    int _161 = 0;
    float _156 = 1.0;
    int _159 = 0;
    _159 = _154[_147];
    _161 = ivec4(1, 2, 3, 4)[_147];
}

void main()
{
    _34();
    _41();
    _46();
    _51();
    _71();
    _77();
    _96();
    _104();
    _110();
    _120();
    _132();
    _104();
    _148(1u);
}

