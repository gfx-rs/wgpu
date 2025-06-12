#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uint sink = 0u;


void simple() {
    uint a = 0u;
    uint _e1 = sink;
    a = _e1;
    uint b = a;
    a = 2u;
    sink = b;
    return;
}

void main() {
    simple();
    return;
}

