#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer type_1_block_0Compute { int _group_0_binding_0_cs[]; };


int simple_for() {
    int sum = 0;
    for(int i = 0; (i < 10); i = (i + 1)) {
        int _e7 = sum;
        int _e8 = i;
        sum = (_e7 + _e8);
    }
    int _e13 = sum;
    return _e13;
}

int simple_while() {
    int i_1 = 0;
    while((i_1 < 10)) {
        int _e6 = i_1;
        i_1 = (_e6 + 1);
    }
    int _e8 = i_1;
    return _e8;
}

int for_with_continue() {
    int sum_1 = 0;
    for(int i_2 = 0; (i_2 < 10); i_2 = (i_2 + 1)) {
        int _e7 = i_2;
        if ((_e7 == 5)) {
            continue;
        }
        int _e10 = sum_1;
        int _e11 = i_2;
        sum_1 = (_e10 + _e11);
    }
    int _e16 = sum_1;
    return _e16;
}

int for_with_break() {
    int sum_2 = 0;
    for(int i_3 = 0; (i_3 < 10); i_3 = (i_3 + 1)) {
        int _e7 = i_3;
        if ((_e7 == 5)) {
            break;
        }
        int _e10 = sum_2;
        int _e11 = i_3;
        sum_2 = (_e10 + _e11);
    }
    int _e16 = sum_2;
    return _e16;
}

int for_infinite() {
    int i_4 = 0;
    for(; ; ) {
        int _e2 = i_4;
        if ((_e2 >= 10)) {
            break;
        }
        int _e6 = i_4;
        i_4 = (_e6 + 1);
    }
    int _e8 = i_4;
    return _e8;
}

int for_no_update() {
    int i_5 = 0;
    for(; (i_5 < 10); ) {
        int _e6 = i_5;
        i_5 = (_e6 + 1);
    }
    int _e8 = i_5;
    return _e8;
}

int while_with_break() {
    int i_6 = 0;
    while(true) {
        int _e3 = i_6;
        if ((_e3 >= 10)) {
            break;
        }
        int _e7 = i_6;
        i_6 = (_e7 + 1);
    }
    int _e9 = i_6;
    return _e9;
}

int nested_loops() {
    int sum_3 = 0;
    int i_7 = 0;
    while((i_7 < 3)) {
        for(int j = 0; (j < 3); j = (j + 1)) {
            int _e12 = sum_3;
            int _e13 = i_7;
            int _e16 = j;
            sum_3 = (_e12 + ((_e13 * 3) + _e16));
        }
        int _e23 = i_7;
        i_7 = (_e23 + 1);
    }
    int _e25 = sum_3;
    return _e25;
}

int for_var_outside() {
    int i_8 = 0;
    for(; (i_8 < 10); i_8 = (i_8 + 1)) {
    }
    int _e8 = i_8;
    return _e8;
}

void main() {
    int _e2 = simple_for();
    _group_0_binding_0_cs[0] = _e2;
    int _e5 = simple_while();
    _group_0_binding_0_cs[1] = _e5;
    int _e8 = for_with_continue();
    _group_0_binding_0_cs[2] = _e8;
    int _e11 = for_with_break();
    _group_0_binding_0_cs[3] = _e11;
    int _e14 = for_infinite();
    _group_0_binding_0_cs[4] = _e14;
    int _e17 = for_no_update();
    _group_0_binding_0_cs[5] = _e17;
    int _e20 = while_with_break();
    _group_0_binding_0_cs[6] = _e20;
    int _e23 = nested_loops();
    _group_0_binding_0_cs[7] = _e23;
    int _e26 = for_var_outside();
    _group_0_binding_0_cs[8] = _e26;
    return;
}

