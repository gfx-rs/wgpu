#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _7()
{
    int _17 = 0;
    uvec2 _19 = uvec2(0u);
    _19 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_19, uvec2(0u))))
        {
            break;
        }
        _19 -= uvec2(uint(_19.y == 0u), 1u);
        if (!(_17 < 4))
        {
            break;
        }
        break;
    }
}

void main()
{
    _7();
}

