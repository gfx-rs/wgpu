#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    vec3 a = cross(vec3(0.0, 1.0, 2.0), vec3(0.0, 1.0, 2.0));
}

