#version 310 es

precision highp float;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared float wg[10];

void main() {
    wg[3] = 1.0;
    return;
}

