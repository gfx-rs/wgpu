#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _6()
{
    uvec2 _22 = uvec2(4294967295u);
    do
    {
        if (all(equal(uvec2(0u), _22)))
        {
            break;
        }
        _22 -= uvec2(uint(_22.y == 0u), 1u);
    } while (!true);
}

void _35(bool _34)
{
    bool _37 = false;
    bool _40 = false;
    uvec2 _47 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _47)))
        {
            break;
        }
        _47 -= uvec2(uint(_47.y == 0u), 1u);
        _37 = _34;
        _40 = _34 != _37;
        if (_34 == _40)
        {
            break;
        }
        else
        {
            continue;
        }
    }
}

void _64(bool _63)
{
    bool _65 = false;
    bool _67 = false;
    uvec2 _74 = uvec2(4294967295u);
    do
    {
        if (all(equal(uvec2(0u), _74)))
        {
            break;
        }
        _74 -= uvec2(uint(_74.y == 0u), 1u);
        _65 = _63;
        _67 = _63 != _65;
    } while (!(_63 == _67));
}

void _90()
{
    uint _92 = 0u;
    uvec2 _99 = uvec2(4294967295u);
    do
    {
        if (all(equal(uvec2(0u), _99)))
        {
            break;
        }
        _99 -= uvec2(uint(_99.y == 0u), 1u);
        _92++;
    } while (!(_92 == 5u));
}

void main()
{
    _6();
    _35(false);
    _64(false);
    _90();
}

