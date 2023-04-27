#version 310 es

precision highp float;
precision highp int;


void fb1_(inout bool cond) {
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e6 = cond;
            bool unnamed = !(_e6);
            if (unnamed) {
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
    fb1_(param);
    return;
}

void main() {
    main_1();
}

