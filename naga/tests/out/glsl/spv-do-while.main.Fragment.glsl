#version 310 es

precision highp float;
precision highp int;


void f_u0028_b1_u003b(inout bool cond) {
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e1 = cond;
            if (!(_e1)) {
                break;
            }
        }
        loop_init = false;
        continue;
    }
    return;
}

void main_1() {
    bool param = false;
    param = false;
    f_u0028_b1_u003b(param);
    return;
}

void main() {
    main_1();
}

