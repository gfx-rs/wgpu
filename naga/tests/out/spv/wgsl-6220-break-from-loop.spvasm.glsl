#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _5()
{
    int _10 = 0;
    uvec2 _27 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _27)))
        {
            break;
        }
        _27 -= uvec2(uint(_27.y == 0u), 1u);
        if (!(_10 < 4))
        {
            break;
        }
        break;
    }
}

void main()
{
    _5();
}

