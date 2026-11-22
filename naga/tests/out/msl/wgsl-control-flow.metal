// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void control_flow(
) {
    int pos = {};
    metal::threadgroup_barrier(metal::mem_flags::mem_device);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_texture);
    switch(1) {
        default: {
            pos = 1;
            break;
        }
    }
    int _e3 = pos;
    switch(_e3) {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            break;
        }
        case 3:
        case 4: {
            pos = 2;
            break;
        }
        case 5: {
            pos = 3;
            break;
        }
        default:
        case 6: {
            pos = 4;
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
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            break;
        }
        case 3: {
            pos = 2;
            break;
        }
        case 4: {
            break;
        }
        default: {
            pos = 3;
            break;
        }
    }
    int _e15 = pos;
    switch(_e15) {
        case 1: {
            pos = 0;
            return;
        }
        case 2: {
            pos = 1;
            return;
        }
        case 3:
        case 4: {
            pos = 2;
            return;
        }
        case 5:
        case 6: {
            pos = 3;
            return;
        }
        default: {
            pos = 4;
            return;
        }
    }
}

void switch_default_break(
    int i
) {
    switch(i) {
        default: {
            break;
        }
    }
}

void switch_case_break(
) {
    switch(0) {
        case 0: {
            break;
        }
        default: {
            break;
        }
    }
    return;
}

void switch_selector_type_conversion(
) {
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

void switch_const_expr_case_selectors(
) {
    switch(0) {
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

void loop_switch_continue(
    int x
) {
    metal::uint2 loop_bound = {};
    loop_bound = metal::uint2(4294967295u);
    while(true) {
        metal::uint2 _e7 = loop_bound;
        if (metal::all(_e7 == metal::uint2(0u))) {
            break;
        }
        loop_bound = _e7 - metal::uint2((_e7.y == 0u) ? 1u : 0u, 1u);
        switch(x) {
            case 1: {
                continue;
            }
            default: {
                break;
            }
        }
    }
    return;
}

void loop_switch_continue_nesting(
    int x_1,
    int y,
    int z
) {
    metal::uint2 loop_bound_1 = {};
    metal::uint2 loop_bound_2 = {};
    metal::uint2 loop_bound_3 = {};
    loop_bound_1 = metal::uint2(4294967295u);
    while(true) {
        metal::uint2 _e24 = loop_bound_1;
        if (metal::all(_e24 == metal::uint2(0u))) {
            break;
        }
        loop_bound_1 = _e24 - metal::uint2((_e24.y == 0u) ? 1u : 0u, 1u);
        switch(x_1) {
            case 1: {
                continue;
            }
            case 2: {
                switch(y) {
                    case 1: {
                        continue;
                    }
                    default: {
                        loop_bound_2 = metal::uint2(4294967295u);
                        while(true) {
                            metal::uint2 _e9 = loop_bound_2;
                            if (metal::all(_e9 == metal::uint2(0u))) {
                                break;
                            }
                            loop_bound_2 = _e9 - metal::uint2((_e9.y == 0u) ? 1u : 0u, 1u);
                            switch(z) {
                                case 1: {
                                    continue;
                                }
                                default: {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
                break;
            }
            default: {
                break;
            }
        }
        switch(y) {
            default: {
                continue;
            }
        }
    }
    loop_bound_3 = metal::uint2(4294967295u);
    while(true) {
        metal::uint2 _e39 = loop_bound_3;
        if (metal::all(_e39 == metal::uint2(0u))) {
            break;
        }
        loop_bound_3 = _e39 - metal::uint2((_e39.y == 0u) ? 1u : 0u, 1u);
        switch(y) {
            case 1:
            default: {
                switch(z) {
                    default: {
                        continue;
                    }
                }
                break;
            }
        }
    }
    return;
}

void loop_switch_omit_continue_variable_checks(
    int x_2,
    int y_1,
    int z_1,
    int w
) {
    int pos_1 = 0;
    metal::uint2 loop_bound_4 = {};
    metal::uint2 loop_bound_5 = {};
    loop_bound_4 = metal::uint2(4294967295u);
    while(true) {
        metal::uint2 _e14 = loop_bound_4;
        if (metal::all(_e14 == metal::uint2(0u))) {
            break;
        }
        loop_bound_4 = _e14 - metal::uint2((_e14.y == 0u) ? 1u : 0u, 1u);
        switch(x_2) {
            case 1: {
                pos_1 = 1;
                break;
            }
            default: {
                break;
            }
        }
    }
    loop_bound_5 = metal::uint2(4294967295u);
    while(true) {
        metal::uint2 _e29 = loop_bound_5;
        if (metal::all(_e29 == metal::uint2(0u))) {
            break;
        }
        loop_bound_5 = _e29 - metal::uint2((_e29.y == 0u) ? 1u : 0u, 1u);
        switch(x_2) {
            case 1: {
                break;
            }
            case 2: {
                switch(y_1) {
                    case 1: {
                        continue;
                    }
                    default: {
                        switch(z_1) {
                            case 1: {
                                pos_1 = 2;
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        break;
                    }
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

kernel void main_(
) {
    control_flow();
    switch_default_break(1);
    switch_case_break();
    switch_selector_type_conversion();
    switch_const_expr_case_selectors();
    loop_switch_continue(1);
    loop_switch_continue_nesting(1, 2, 3);
    loop_switch_omit_continue_variable_checks(1, 2, 3, 4);
    return;
}
