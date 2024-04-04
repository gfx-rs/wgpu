#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const bool has_point_light = false;
const float specular_param = 2.3;
const float gain = 1.1;
const float width = 0.0;
const float depth = 2.3;
const float height = 4.6;
const float inferred_f32_ = 2.718;

float gain_x_10_ = 11.0;


void main() {
    float t = 0.0;
    bool x = false;
    float gain_x_100_ = 0.0;
    t = 23.0;
    x = true;
    float _e10 = gain_x_10_;
    gain_x_100_ = (_e10 * 10.0);
    return;
}

