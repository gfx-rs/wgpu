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
    uvec2 _88 = uvec2(0u);
    _88 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_88, uvec2(0u))))
        {
            break;
        }
        _88 -= uvec2(uint(_88.y == 0u), 1u);
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

void _115(int _112, int _113, int _114)
{
    uvec2 _117 = uvec2(0u);
    uvec2 _119 = uvec2(0u);
    uvec2 _121 = uvec2(0u);
    _117 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_117, uvec2(0u))))
        {
            break;
        }
        _117 -= uvec2(uint(_117.y == 0u), 1u);
        switch (_112)
        {
            case 1:
            {
                continue;
            }
            case 2:
            {
                switch (_113)
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
                            if (all(equal(_119, uvec2(0u))))
                            {
                                break;
                            }
                            _119 -= uvec2(uint(_119.y == 0u), 1u);
                            switch (_114)
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
    _121 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_121, uvec2(0u))))
        {
            break;
        }
        _121 -= uvec2(uint(_121.y == 0u), 1u);
        switch (_113)
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

void _187(int _183, int _184, int _185, int _186)
{
    int _189 = 0;
    uvec2 _190 = uvec2(0u);
    uvec2 _192 = uvec2(0u);
    _190 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_190, uvec2(0u))))
        {
            break;
        }
        _190 -= uvec2(uint(_190.y == 0u), 1u);
        switch (_183)
        {
            case 1:
            {
                _189 = 1;
                break;
            }
            default:
            {
                break;
            }
        }
        continue;
    }
    _192 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_192, uvec2(0u))))
        {
            break;
        }
        _192 -= uvec2(uint(_192.y == 0u), 1u);
        switch (_183)
        {
            case 1:
            {
                break;
            }
            case 2:
            {
                switch (_184)
                {
                    case 1:
                    {
                        continue;
                    }
                    default:
                    {
                        switch (_185)
                        {
                            case 1:
                            {
                                _189 = 2;
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
    _115(1, 2, 3);
    _187(1, 2, 3, 4);
}

