#version 460
layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;

const int _155[9] = int[](1, 2, 3, 4, 5, 6, 7, 8, 9);

void _35()
{
    ivec4 _38 = ivec4(4, 3, 2, 1);
}

void _42()
{
    int _43 = 2;
}

void _47()
{
    int _49 = 6;
}

void _52()
{
    ivec4 _61 = ivec4(0);
    int _56 = 0;
    int _60 = 70;
    int _55 = 30;
    int _58 = 0;
    _56 = _55;
    _58 = _56;
    _61 = ivec4(_55, _56, _58, _60);
}

void _72()
{
    ivec4 _75 = ivec4(-4);
}

void _78()
{
    ivec4 _79 = ivec4(-4);
}

uint _83(int _82)
{
    switch (_82)
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

void _97()
{
    vec4 _101 = vec4(2.0, 1.0, 1.0, 1.0);
}

void _105()
{
    float _106[2] = float[](0.0, 0.0);
}

void _111()
{
    ivec3 _115 = ivec3(1);
    ivec3 _117 = ivec3(0, 1, 2);
    ivec3 _118 = ivec3(1, 0, 2);
}

void _121()
{
    bool _129 = false;
    bool _126 = true;
    bool _122 = false;
    bool _130 = true;
    bool _127 = false;
    bool _124 = true;
    bool _128 = true;
    bool _125 = false;
}

void _133()
{
    int _143 = 70;
    uint _139 = 4u;
    int _145 = -4;
    uint _142 = 12u;
    int _138 = 4;
    uint _144 = 70u;
    int _141 = 12;
}

void _149(uint _148)
{
    uint _159 = 1u;
    int _162 = 0;
    float _157 = 1.0;
    int _160 = 0;
    _160 = _155[_148];
    _162 = ivec4(1, 2, 3, 4)[_148];
}

void _171()
{
    ivec2 _173 = ivec2(0, 3);
}

void _177()
{
    int _178 = 0;
}

void _181()
{
    int _182 = 0;
}

void _185()
{
    ivec3 _187 = ivec3(0);
}

void main()
{
    _35();
    _42();
    _47();
    _52();
    _72();
    _78();
    _97();
    _105();
    _111();
    _121();
    _133();
    _105();
    _149(1u);
    _171();
    _177();
    _181();
    _185();
}

