void control_flow()
{
    int pos = (int)0;

    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();
    do {
        pos = int(1);
    } while(false);
    int _e3 = pos;
    switch(_e3) {
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
    int _e10 = pos;
    switch(_e10) {
        case 1: {
            pos = int(0);
            break;
        }
        case 2: {
            pos = int(1);
            return;
        }
        case 3: {
            pos = int(2);
            return;
        }
        case 4: {
            return;
        }
        default: {
            pos = int(3);
            return;
        }
    }
}

void switch_default_break(int i)
{
    do {
        break;
    } while(false);
}

void switch_case_break()
{
    switch(int(0)) {
        case 0: {
            break;
        }
        default: {
            break;
        }
    }
    return;
}

void switch_selector_type_conversion()
{
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
            return;
        }
        default: {
            return;
        }
    }
}

void switch_const_expr_case_selectors()
{
    switch(int(0)) {
        case 0: {
            return;
        }
        case 1: {
            return;
        }
        case 2: {
            return;
        }
        case 3: {
            return;
        }
        case 4: {
            return;
        }
        default: {
            return;
        }
    }
}

void loop_switch_continue(int x)
{
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool should_continue = false;
        switch(x) {
            case 1: {
                should_continue = true;
                break;
            }
            default: {
                break;
            }
        }
        if (should_continue) {
            continue;
        }
    }
    return;
}

void loop_switch_continue_nesting(int x_1, int y, int z)
{
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        bool should_continue_1 = false;
        switch(x_1) {
            case 1: {
                should_continue_1 = true;
                break;
            }
            case 2: {
                switch(y) {
                    case 1: {
                        should_continue_1 = true;
                        break;
                    }
                    default: {
                        uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
                        while(true) {
                            if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
                            loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
                            bool should_continue_2 = false;
                            switch(z) {
                                case 1: {
                                    should_continue_2 = true;
                                    break;
                                }
                                default: {
                                    break;
                                }
                            }
                            if (should_continue_2) {
                                continue;
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
        if (should_continue_3) {
            continue;
        }
    }
    uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
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
        if (should_continue_4) {
            continue;
        }
    }
    return;
}

void loop_switch_omit_continue_variable_checks(int x_2, int y_1, int z_1, int w)
{
    int pos_1 = int(0);

    uint2 loop_bound_4 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_4 == uint2(0u, 0u))) { break; }
        loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
        bool should_continue_5 = false;
        switch(x_2) {
            case 1: {
                pos_1 = int(1);
                break;
            }
            default: {
                break;
            }
        }
    }
    uint2 loop_bound_5 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_5 == uint2(0u, 0u))) { break; }
        loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
        bool should_continue_6 = false;
        switch(x_2) {
            case 1: {
                break;
            }
            case 2: {
                switch(y_1) {
                    case 1: {
                        should_continue_6 = true;
                        break;
                    }
                    default: {
                        switch(z_1) {
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
        if (should_continue_6) {
            continue;
        }
    }
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    control_flow();
    switch_default_break(int(1));
    switch_case_break();
    switch_selector_type_conversion();
    switch_const_expr_case_selectors();
    loop_switch_continue(int(1));
    loop_switch_continue_nesting(int(1), int(2), int(3));
    loop_switch_omit_continue_variable_checks(int(1), int(2), int(3), int(4));
    return;
}
