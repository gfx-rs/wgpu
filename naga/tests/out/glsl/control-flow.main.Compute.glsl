#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void switch_default_break(int i) {
    do {
        break;
    } while(false);
}

void switch_case_break() {
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

void loop_switch_continue(int x) {
    while(true) {
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

void loop_switch_continue_nesting(int x_1, int y, int z) {
    while(true) {
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
                        while(true) {
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
        bool should_continue = false;
        do {
            should_continue = true;
            break;
        } while(false);
        if (should_continue) {
            continue;
        }
    }
    while(true) {
        bool should_continue_1 = false;
        do {
            do {
                should_continue_1 = true;
                break;
            } while(false);
            if (should_continue_1) {
                break;
            }
        } while(false);
        if (should_continue_1) {
            continue;
        }
    }
    return;
}

void loop_switch_omit_continue_variable_checks(int x_2, int y_1, int z_1, int w) {
    int pos_1 = 0;
    while(true) {
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
    while(true) {
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

void main() {
    uvec3 global_id = gl_GlobalInvocationID;
    int pos = 0;
    memoryBarrierBuffer();
    barrier();
    memoryBarrierShared();
    barrier();
    do {
        pos = 1;
    } while(false);
    int _e4 = pos;
    switch(_e4) {
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
    int _e11 = pos;
    switch(_e11) {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            return;
        }
        case 3: {
            pos = 2;
            return;
        }
        case 4: {
            return;
        }
        default: {
            pos = 3;
            return;
        }
    }
}

