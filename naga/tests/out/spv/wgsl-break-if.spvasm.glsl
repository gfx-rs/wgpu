#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _7()
{
    uvec2 _15 = uvec2(4294967295u);
    do
    {
        uvec2 _24 = _15;
        _15 = _24 - uvec2(uint(_24.y == 0u), 1u);
        if (all(equal(_24, uvec2(0u))))
        {
            break;
        }
    } while (!true);
}

void _37(bool _36)
{
    bool _39 = false;
    bool _42 = false;
    uvec2 _44 = uvec2(4294967295u);
    for (;;)
    {
        uvec2 _52 = _44;
        _44 = _52 - uvec2(uint(_52.y == 0u), 1u);
        if (all(equal(_52, uvec2(0u))))
        {
            break;
        }
        _39 = _36;
        _42 = _36 != _39;
        if (_36 == _42)
        {
            break;
        }
        else
        {
            continue;
        }
    }
}

void _68(bool _67)
{
    bool _69 = false;
    bool _71 = false;
    uvec2 _73 = uvec2(4294967295u);
    do
    {
        uvec2 _81 = _73;
        _73 = _81 - uvec2(uint(_81.y == 0u), 1u);
        if (all(equal(_81, uvec2(0u))))
        {
            break;
        }
        _69 = _67;
        _71 = _67 != _69;
    } while (!(_67 == _71));
}

void _96()
{
    uint _98 = 0u;
    uvec2 _100 = uvec2(4294967295u);
    do
    {
        uvec2 _108 = _100;
        _100 = _108 - uvec2(uint(_108.y == 0u), 1u);
        if (all(equal(_108, uvec2(0u))))
        {
            break;
        }
        _98++;
    } while (!(_98 == 5u));
}

void main()
{
    _7();
    _37(false);
    _68(false);
    _96();
}

