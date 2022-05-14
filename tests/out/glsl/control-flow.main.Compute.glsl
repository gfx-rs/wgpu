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
        default:
            break;
    }
    return;
}

void loop_switch_continue(int x) {
    while(true) {
        switch(x) {
            case 1:
                continue;
            default:
                break;
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
    switch(1) {
        default:
            pos = 1;
            break;
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
            pos = 3;
            /* fallthrough */
        default:
            pos = 4;
            break;
    }
    switch(0u) {
        case 0u:
            break;
        default:
            break;
    }
    int _e11 = pos;
    switch(_e11) {
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

