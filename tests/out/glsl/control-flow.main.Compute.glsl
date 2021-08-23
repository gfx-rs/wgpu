#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


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
            return;
        case 3:
            pos = 2;
        case 4:
            return;
        default:
            pos = 3;
            return;
    }
}

