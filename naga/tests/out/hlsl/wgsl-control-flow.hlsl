int ZeroValueint() {
    return (int)0;
}

[numthreads(1, 1, 1)]
void main()
{
    int pos = (int)0;
    bool local = (bool)0;
    int i = (int)0;
    bool local_1 = (bool)0;
    bool local_2 = (bool)0;
    int x = (int)0;
    int x_1 = (int)0;
    int y = (int)0;
    int z = (int)0;
    int x_2 = (int)0;
    int y_1 = (int)0;
    int z_1 = (int)0;
    int w = (int)0;
    int pos_1 = (int)0;
    uint2 loop_bound = (4294967295u).xx;
    uint2 loop_bound_1 = (4294967295u).xx;
    uint2 loop_bound_2 = (4294967295u).xx;
    uint2 loop_bound_3 = (4294967295u).xx;
    uint2 loop_bound_4 = (4294967295u).xx;
    uint2 loop_bound_5 = (4294967295u).xx;

    do {
        pos = ZeroValueint();
        local = false;
        DeviceMemoryBarrierWithGroupSync();
        GroupMemoryBarrierWithGroupSync();
        DeviceMemoryBarrierWithGroupSync();
        do {
            pos = int(1);
        } while(false);
        int _e12 = pos;
        switch(_e12) {
            case 1: {
                pos = int(0);
                break;
            }
            case 2: {
                pos = int(1);
                break;
            }
            case 3:
            case 4: {
                pos = int(2);
                break;
            }
            case 5: {
                pos = int(3);
                break;
            }
            default:
            case 6: {
                pos = int(4);
                break;
            }
        }
        switch(0u) {
            case 0u: {
                break;
            }
            default: {
                break;
            }
        }
        int _e19 = pos;
        switch(_e19) {
            case 1: {
                pos = int(0);
                break;
            }
            case 2: {
                pos = int(1);
                break;
            }
            case 3: {
                pos = int(2);
                break;
            }
            case 4: {
                break;
            }
            default: {
                pos = int(3);
                break;
            }
        }
        int _e24 = pos;
        switch(_e24) {
            case 1: {
                pos = int(0);
                local = true;
                break;
            }
            case 2: {
                pos = int(1);
                local = true;
                break;
            }
            case 3:
            case 4: {
                pos = int(2);
                local = true;
                break;
            }
            case 5:
            case 6: {
                pos = int(3);
                local = true;
                break;
            }
            default: {
                pos = int(4);
                local = true;
                break;
            }
        }
        bool _e35 = local;
        if (_e35) {
            break;
        }
    } while(false);
    i = int(1);
    int _e38 = i;
    do {
        break;
    } while(false);
    switch(int(0)) {
        case 0: {
            break;
        }
        default: {
            break;
        }
    }
    do {
        local_1 = false;
        switch(0u) {
            case 0u: {
                break;
            }
            default: {
                break;
            }
        }
        switch(0u) {
            case 0u: {
                local_1 = true;
                break;
            }
            default: {
                local_1 = true;
                break;
            }
        }
        bool _e45 = local_1;
        if (_e45) {
            break;
        }
    } while(false);
    do {
        local_2 = false;
        switch(int(0)) {
            case 0: {
                local_2 = true;
                break;
            }
            case 1: {
                local_2 = true;
                break;
            }
            case 2: {
                local_2 = true;
                break;
            }
            case 3: {
                local_2 = true;
                break;
            }
            case 4: {
                local_2 = true;
                break;
            }
            default: {
                local_2 = true;
                break;
            }
        }
        bool _e51 = local_2;
        if (_e51) {
            break;
        }
    } while(false);
    x = int(1);
    int _e54 = x;
    while(true) {
        {
            uint2 _e78 = loop_bound;
            loop_bound = (_e78 - uint2(((_e78.y == 0u) ? 1u : 0u), 1u));
            if (all((_e78 == (0u).xx))) {
                break;
            }
        }
        bool should_continue = false;
        switch(_e54) {
            case 1: {
                should_continue = true;
                break;
            }
            default: {
                break;
            }
        }
    }
    x_1 = int(1);
    int _e56 = x_1;
    y = int(2);
    int _e58 = y;
    z = int(3);
    int _e60 = z;
    while(true) {
        {
            uint2 _e105 = loop_bound_1;
            loop_bound_1 = (_e105 - uint2(((_e105.y == 0u) ? 1u : 0u), 1u));
            if (all((_e105 == (0u).xx))) {
                break;
            }
        }
        bool should_continue_1 = false;
        switch(_e56) {
            case 1: {
                should_continue_1 = true;
                break;
            }
            case 2: {
                switch(_e58) {
                    case 1: {
                        should_continue_1 = true;
                        break;
                    }
                    default: {
                        loop_bound_2 = (4294967295u).xx;
                        while(true) {
                            {
                                uint2 _e93 = loop_bound_2;
                                loop_bound_2 = (_e93 - uint2(((_e93.y == 0u) ? 1u : 0u), 1u));
                                if (all((_e93 == (0u).xx))) {
                                    break;
                                }
                            }
                            bool should_continue_2 = false;
                            switch(_e60) {
                                case 1: {
                                    should_continue_2 = true;
                                    break;
                                }
                                default: {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
                if (should_continue_1) {
                    break;
                }
                break;
            }
            default: {
                break;
            }
        }
        if (should_continue_1) {
            continue;
        }
        bool should_continue_3 = false;
        do {
            should_continue_3 = true;
            break;
        } while(false);
    }
    while(true) {
        {
            uint2 _e117 = loop_bound_3;
            loop_bound_3 = (_e117 - uint2(((_e117.y == 0u) ? 1u : 0u), 1u));
            if (all((_e117 == (0u).xx))) {
                break;
            }
        }
        bool should_continue_4 = false;
        do {
            do {
                should_continue_4 = true;
                break;
            } while(false);
            if (should_continue_4) {
                break;
            }
        } while(false);
    }
    x_2 = int(1);
    int _e62 = x_2;
    y_1 = int(2);
    int _e64 = y_1;
    z_1 = int(3);
    int _e66 = z_1;
    w = int(4);
    pos_1 = int(0);
    while(true) {
        {
            uint2 _e129 = loop_bound_4;
            loop_bound_4 = (_e129 - uint2(((_e129.y == 0u) ? 1u : 0u), 1u));
            if (all((_e129 == (0u).xx))) {
                break;
            }
        }
        bool should_continue_5 = false;
        switch(_e62) {
            case 1: {
                pos_1 = int(1);
                break;
            }
            default: {
                break;
            }
        }
    }
    while(true) {
        {
            uint2 _e141 = loop_bound_5;
            loop_bound_5 = (_e141 - uint2(((_e141.y == 0u) ? 1u : 0u), 1u));
            if (all((_e141 == (0u).xx))) {
                break;
            }
        }
        bool should_continue_6 = false;
        switch(_e62) {
            case 1: {
                break;
            }
            case 2: {
                switch(_e64) {
                    case 1: {
                        should_continue_6 = true;
                        break;
                    }
                    default: {
                        switch(_e66) {
                            case 1: {
                                pos_1 = int(2);
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        break;
                    }
                }
                if (should_continue_6) {
                    break;
                }
                break;
            }
            default: {
                break;
            }
        }
    }
    return;
}
