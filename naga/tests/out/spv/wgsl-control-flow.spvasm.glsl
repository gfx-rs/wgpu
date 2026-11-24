#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _7()
{
    int _15 = 0;
    memoryBarrierBuffer();
    barrier();
    barrier();
    groupMemoryBarrier();
    barrier();
    do
    {
        _15 = 1;
        break;
    } while(false);
    switch (_15)
    {
        case 1:
        {
            _15 = 0;
            break;
        }
        case 2:
        {
            _15 = 1;
            break;
        }
        case 3:
        case 4:
        {
            _15 = 2;
            break;
        }
        case 5:
        {
            _15 = 3;
            break;
        }
        default:
        {
            _15 = 4;
            break;
        }
    }
    switch (0u)
    {
        case 0u:
        {
            break;
        }
        default:
        {
            break;
        }
    }
    switch (_15)
    {
        case 1:
        {
            _15 = 0;
            break;
        }
        case 2:
        {
            _15 = 1;
            break;
        }
        case 3:
        {
            _15 = 2;
            break;
        }
        case 4:
        {
            break;
        }
        default:
        {
            _15 = 3;
            break;
        }
    }
    switch (_15)
    {
        case 1:
        {
            _15 = 0;
            return;
        }
        case 2:
        {
            _15 = 1;
            return;
        }
        case 3:
        case 4:
        {
            _15 = 2;
            return;
        }
        case 5:
        case 6:
        {
            _15 = 3;
            return;
        }
        default:
        {
            _15 = 4;
            return;
        }
    }
}

void _52(int _51)
{
    do
    {
        break;
    } while(false);
}

void _58()
{
    switch (0)
    {
        case 0:
        {
            break;
        }
        default:
        {
            break;
        }
    }
}

void _64()
{
    switch (0u)
    {
        case 0u:
        {
            break;
        }
        default:
        {
            break;
        }
    }
    switch (0u)
    {
        case 0u:
        {
            return;
        }
        default:
        {
            return;
        }
    }
}

void _73()
{
    switch (0)
    {
        case 0:
        {
            return;
        }
        case 1:
        {
            return;
        }
        case 2:
        {
            return;
        }
        case 3:
        {
            return;
        }
        case 4:
        {
            return;
        }
        default:
        {
            return;
        }
    }
}

void _84(int _83)
{
    uvec2 _88 = uvec2(4294967295u);
    for (;;)
    {
        uvec2 _97 = _88;
        _88 = _97 - uvec2(uint(_97.y == 0u), 1u);
        if (all(equal(_97, uvec2(0u))))
        {
            break;
        }
        switch (_83)
        {
            case 1:
            {
                continue;
            }
            default:
            {
                break;
            }
        }
        continue;
    }
}

void _116(int _113, int _114, int _115)
{
    uvec2 _118 = uvec2(4294967295u);
    uvec2 _119 = uvec2(4294967295u);
    uvec2 _120 = uvec2(4294967295u);
    for (;;)
    {
        uvec2 _128 = _118;
        _118 = _128 - uvec2(uint(_128.y == 0u), 1u);
        if (all(equal(_128, uvec2(0u))))
        {
            break;
        }
        switch (_113)
        {
            case 1:
            {
                continue;
            }
            case 2:
            {
                switch (_114)
                {
                    case 1:
                    {
                        continue;
                    }
                    default:
                    {
                        _119 = uvec2(4294967295u);
                        for (;;)
                        {
                            uvec2 _151 = _119;
                            _119 = _151 - uvec2(uint(_151.y == 0u), 1u);
                            if (all(equal(_151, uvec2(0u))))
                            {
                                break;
                            }
                            switch (_115)
                            {
                                case 1:
                                {
                                    continue;
                                }
                                default:
                                {
                                    break;
                                }
                            }
                            continue;
                        }
                        break;
                    }
                }
                break;
            }
            default:
            {
                break;
            }
        }
        do
        {
            continue;
        } while(false);
        continue;
    }
    for (;;)
    {
        uvec2 _172 = _120;
        _120 = _172 - uvec2(uint(_172.y == 0u), 1u);
        if (all(equal(_172, uvec2(0u))))
        {
            break;
        }
        switch (_114)
        {
            default:
            {
                do
                {
                    continue;
                } while(false);
                break;
            }
        }
        continue;
    }
}

void _191(int _187, int _188, int _189, int _190)
{
    int _193 = 0;
    uvec2 _194 = uvec2(4294967295u);
    uvec2 _195 = uvec2(4294967295u);
    for (;;)
    {
        uvec2 _203 = _194;
        _194 = _203 - uvec2(uint(_203.y == 0u), 1u);
        if (all(equal(_203, uvec2(0u))))
        {
            break;
        }
        switch (_187)
        {
            case 1:
            {
                _193 = 1;
                break;
            }
            default:
            {
                break;
            }
        }
        continue;
    }
    for (;;)
    {
        uvec2 _222 = _195;
        _195 = _222 - uvec2(uint(_222.y == 0u), 1u);
        if (all(equal(_222, uvec2(0u))))
        {
            break;
        }
        switch (_187)
        {
            case 1:
            {
                break;
            }
            case 2:
            {
                switch (_188)
                {
                    case 1:
                    {
                        continue;
                    }
                    default:
                    {
                        switch (_189)
                        {
                            case 1:
                            {
                                _193 = 2;
                                break;
                            }
                            default:
                            {
                                break;
                            }
                        }
                        break;
                    }
                }
                break;
            }
            default:
            {
                break;
            }
        }
        continue;
    }
}

void main()
{
    _7();
    _52(1);
    _58();
    _64();
    _73();
    _84(1);
    _116(1, 2, 3);
    _191(1, 2, 3, 4);
}

