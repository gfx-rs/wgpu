#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _5()
{
    int _14 = 0;
    memoryBarrierBuffer();
    barrier();
    barrier();
    groupMemoryBarrier();
    barrier();
    do
    {
        _14 = 1;
        break;
    } while(false);
    switch (_14)
    {
        case 1:
        {
            _14 = 0;
            break;
        }
        case 2:
        {
            _14 = 1;
            break;
        }
        case 3:
        case 4:
        {
            _14 = 2;
            break;
        }
        case 5:
        {
            _14 = 3;
            break;
        }
        default:
        {
            _14 = 4;
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
    switch (_14)
    {
        case 1:
        {
            _14 = 0;
            break;
        }
        case 2:
        {
            _14 = 1;
            break;
        }
        case 3:
        {
            _14 = 2;
            break;
        }
        case 4:
        {
            break;
        }
        default:
        {
            _14 = 3;
            break;
        }
    }
    switch (_14)
    {
        case 1:
        {
            _14 = 0;
            return;
        }
        case 2:
        {
            _14 = 1;
            return;
        }
        case 3:
        case 4:
        {
            _14 = 2;
            return;
        }
        case 5:
        case 6:
        {
            _14 = 3;
            return;
        }
        default:
        {
            _14 = 4;
            return;
        }
    }
}

void _51(int _50)
{
    do
    {
        break;
    } while(false);
}

void _57()
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

void _63()
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

void _72()
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

void _83(int _82)
{
    uvec2 _96 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _96)))
        {
            break;
        }
        _96 -= uvec2(uint(_96.y == 0u), 1u);
        switch (_82)
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

void _114(int _111, int _112, int _113)
{
    uvec2 _121 = uvec2(4294967295u);
    uvec2 _143 = uvec2(4294967295u);
    uvec2 _163 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _121)))
        {
            break;
        }
        _121 -= uvec2(uint(_121.y == 0u), 1u);
        switch (_111)
        {
            case 1:
            {
                continue;
            }
            case 2:
            {
                switch (_112)
                {
                    case 1:
                    {
                        continue;
                    }
                    default:
                    {
                        for (;;)
                        {
                            if (all(equal(uvec2(0u), _143)))
                            {
                                break;
                            }
                            _143 -= uvec2(uint(_143.y == 0u), 1u);
                            switch (_113)
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
        if (all(equal(uvec2(0u), _163)))
        {
            break;
        }
        _163 -= uvec2(uint(_163.y == 0u), 1u);
        switch (_112)
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

void _183(int _179, int _180, int _181, int _182)
{
    int _185 = 0;
    uvec2 _191 = uvec2(4294967295u);
    uvec2 _209 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _191)))
        {
            break;
        }
        _191 -= uvec2(uint(_191.y == 0u), 1u);
        switch (_179)
        {
            case 1:
            {
                _185 = 1;
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
        if (all(equal(uvec2(0u), _209)))
        {
            break;
        }
        _209 -= uvec2(uint(_209.y == 0u), 1u);
        switch (_179)
        {
            case 1:
            {
                break;
            }
            case 2:
            {
                switch (_180)
                {
                    case 1:
                    {
                        continue;
                    }
                    default:
                    {
                        switch (_181)
                        {
                            case 1:
                            {
                                _185 = 2;
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
    _5();
    _51(1);
    _57();
    _63();
    _72();
    _83(1);
    _114(1, 2, 3);
    _183(1, 2, 3, 4);
}

