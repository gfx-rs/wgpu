#version 420 core
#extension GL_ARB_compute_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const double k = 2.0LF;


double f(double x) {
    double z = 0.0;
    double y = (30.0LF + 400.0LF);
    z = (y + 5.0LF);
    return (((x + y) + k) + 5.0LF);
}

void main() {
    double _e1 = f(6.0LF);
    return;
}

