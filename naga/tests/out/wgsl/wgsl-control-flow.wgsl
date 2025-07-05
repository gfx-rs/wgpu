fn control_flow() {
    var pos: i32;

    storageBarrier();
    workgroupBarrier();
    textureBarrier();
    switch 1i {
        default: {
            pos = 1i;
        }
    }
    let _e3 = pos;
    switch _e3 {
        case 1: {
            pos = 0i;
            break;
        }
        case 2: {
            pos = 1i;
        }
        case 3, 4: {
            pos = 2i;
        }
        case 5: {
            pos = 3i;
        }
        case default, 6: {
            pos = 4i;
        }
    }
    switch 0u {
        case 0u: {
        }
        default: {
        }
    }
    let _e10 = pos;
    switch _e10 {
        case 1: {
            pos = 0i;
            break;
        }
        case 2: {
            pos = 1i;
            return;
        }
        case 3: {
            pos = 2i;
            return;
        }
        case 4: {
            return;
        }
        default: {
            pos = 3i;
            return;
        }
    }
}

fn switch_default_break(i: i32) {
    switch i {
        default: {
            break;
        }
    }
}

fn switch_case_break() {
    switch 0i {
        case 0: {
            break;
        }
        default: {
        }
    }
    return;
}

fn switch_selector_type_conversion() {
    switch 0u {
        case 0u: {
        }
        default: {
        }
    }
    switch 0u {
        case 0u: {
            return;
        }
        default: {
            return;
        }
    }
}

fn switch_const_expr_case_selectors() {
    switch 0i {
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

fn loop_switch_continue(x: i32) {
    loop {
        switch x {
            case 1: {
                continue;
            }
            default: {
            }
        }
    }
    return;
}

fn loop_switch_continue_nesting(x_1: i32, y: i32, z: i32) {
    loop {
        switch x_1 {
            case 1: {
                continue;
            }
            case 2: {
                switch y {
                    case 1: {
                        continue;
                    }
                    default: {
                        loop {
                            switch z {
                                case 1: {
                                    continue;
                                }
                                default: {
                                }
                            }
                        }
                    }
                }
            }
            default: {
            }
        }
        switch y {
            default: {
                continue;
            }
        }
    }
    loop {
        switch y {
            case 1, default: {
                switch z {
                    default: {
                        continue;
                    }
                }
            }
        }
    }
    return;
}

fn loop_switch_omit_continue_variable_checks(x_2: i32, y_1: i32, z_1: i32, w: i32) {
    var pos_1: i32 = 0i;

    loop {
        switch x_2 {
            case 1: {
                pos_1 = 1i;
            }
            default: {
            }
        }
    }
    loop {
        switch x_2 {
            case 1: {
            }
            case 2: {
                switch y_1 {
                    case 1: {
                        continue;
                    }
                    default: {
                        switch z_1 {
                            case 1: {
                                pos_1 = 2i;
                            }
                            default: {
                            }
                        }
                    }
                }
            }
            default: {
            }
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    control_flow();
    switch_default_break(1i);
    switch_case_break();
    switch_selector_type_conversion();
    switch_const_expr_case_selectors();
    loop_switch_continue(1i);
    loop_switch_continue_nesting(1i, 2i, 3i);
    loop_switch_omit_continue_variable_checks(1i, 2i, 3i, 4i);
    return;
}
