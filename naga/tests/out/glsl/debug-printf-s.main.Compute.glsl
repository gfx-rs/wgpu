#version 310 es
#extension GL_EXT_debug_printf : enable

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main_1() {
    debugPrintfEXT("%d",42);
    return;
}

void main() {
    main_1();
}

