#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    int phony = ivec4(bitfieldExtract(int(12u), 0, 8), bitfieldExtract(int(12u), 8, 8), bitfieldExtract(int(12u), 16, 8), bitfieldExtract(int(12u), 24, 8))[2];
    uint phony_1 = uvec4(bitfieldExtract(12u, 0, 8), bitfieldExtract(12u, 8, 8), bitfieldExtract(12u, 16, 8), bitfieldExtract(12u, 24, 8)).y;
}

