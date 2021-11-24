#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared float wg[10];

shared uint at_1;


void main() {
    float Foo = 1.0;
    bool at = true;
    wg[3] = 1.0;
    at_1 = 2u;
    return;
}

