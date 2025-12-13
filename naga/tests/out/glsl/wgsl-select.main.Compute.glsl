#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    ivec2 x0_ = ivec2(1, 2);
    vec2 i1_ = vec2(0.0);
    int _e12 = x0_.x;
    int _e14 = x0_.y;
    i1_ = ((_e12 < _e14) ? vec2(0.0, 1.0) : vec2(1.0, 0.0));
    return;
}

