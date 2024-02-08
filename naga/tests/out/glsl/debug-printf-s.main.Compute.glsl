#version 450 core
#extension GL_ARB_compute_shader : require
#extension GL_EXT_debug_printf : enable
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main_1() {
    debugPrintfEXT("%d",42);
    return;
}

void main() {
    main_1();
}

