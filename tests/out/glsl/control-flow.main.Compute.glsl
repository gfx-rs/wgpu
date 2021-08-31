#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void switch_default_break(int i) {
    switch(i) {
        default:
            break;
    }
}

void switch_case_break() {
    switch(0) {
        case 0:
            break;
    }
    return;
}

void loop_switch_continue(int x) {
    while(true) {
        switch(x) {
            case 1:
                continue;
        }
    }
    return;
}

void main() {
    uvec3 global_id = gl_GlobalInvocationID;
    int pos = 0;
    groupMemoryBarrier();
    groupMemoryBarrier();
    switch(1) {
        default:
            pos = 1;
    }
    int _e4 = pos;
    switch(_e4) {
        case 1:
            pos = 0;
            break;
        case 2:
            pos = 1;
            break;
        case 3:
            pos = 2;
            /* fallthrough */
        case 4:
            break;
        default:
            pos = 3;
    }
    int _e9 = pos;
    switch(_e9) {
        case 1:
            pos = 0;
            break;
        case 2:
            pos = 1;
            return;
        case 3:
            pos = 2;
            /* fallthrough */
        case 4:
            return;
        default:
            pos = 3;
            return;
    }
}

