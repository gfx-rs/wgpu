#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _7()
{
    uvec2 _15 = uvec2(0u);
    _15 = uvec2(4294967295u);
    do
    {
        if (all(equal(_15, uvec2(0u))))
        {
            break;
        }
        _15 -= uvec2(uint(_15.y == 0u), 1u);
    } while (!true);
}

void _36(bool _35)
{
    bool _38 = false;
    bool _41 = false;
    uvec2 _43 = uvec2(0u);
    _43 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_43, uvec2(0u))))
        {
            break;
        }
        _43 -= uvec2(uint(_43.y == 0u), 1u);
        _38 = _35;
        _41 = _35 != _38;
        if (_35 == _41)
        {
            break;
        }
        else
        {
            continue;
        }
    }
}

void _66(bool _65)
{
    bool _67 = false;
    bool _69 = false;
    uvec2 _71 = uvec2(0u);
    _71 = uvec2(4294967295u);
    do
    {
        if (all(equal(_71, uvec2(0u))))
        {
            break;
        }
        _71 -= uvec2(uint(_71.y == 0u), 1u);
        _67 = _65;
        _69 = _65 != _67;
    } while (!(_65 == _69));
}

void _93()
{
    uint _95 = 0u;
    uvec2 _97 = uvec2(0u);
    _97 = uvec2(4294967295u);
    do
    {
        if (all(equal(_97, uvec2(0u))))
        {
            break;
        }
        _97 -= uvec2(uint(_97.y == 0u), 1u);
        _95++;
    } while (!(_95 == 5u));
}

void main()
{
    _7();
    _36(false);
    _66(false);
    _93();
}

