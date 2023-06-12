#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void breakIfEmpty() {
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            if (true) {
                break;
            }
        }
        loop_init = false;
    }
    return;
}

void breakIfEmptyBody(bool a) {
    bool b = false;
    bool c = false;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            b = a;
            bool _e2 = b;
            c = (a != _e2);
            bool _e5 = c;
            if ((a == _e5)) {
                break;
            }
        }
        loop_init_1 = false;
    }
    return;
}

void breakIf(bool a_1) {
    bool d = false;
    bool e = false;
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            bool _e5 = e;
            if ((a_1 == _e5)) {
                break;
            }
        }
        loop_init_2 = false;
        d = a_1;
        bool _e2 = d;
        e = (a_1 != _e2);
    }
    return;
}

void main() {
    return;
}

