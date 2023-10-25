#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


vec2 test_fma() {
    vec2 a = vec2(2.0, 2.0);
    vec2 b = vec2(0.5, 0.5);
    vec2 c = vec2(0.5, 0.5);
    return (a * b + c);
}

int test_integer_dot_product() {
    ivec2 a_2_ = ivec2(1);
    ivec2 b_2_ = ivec2(1);
    int c_2_ = ( + a_2_.x * b_2_.x + a_2_.y * b_2_.y);
    uvec3 a_3_ = uvec3(1u);
    uvec3 b_3_ = uvec3(1u);
    uint c_3_ = ( + a_3_.x * b_3_.x + a_3_.y * b_3_.y + a_3_.z * b_3_.z);
    ivec4 _e11 = ivec4(4);
    ivec4 _e13 = ivec4(2);
    int c_4_ = ( + _e11.x * _e13.x + _e11.y * _e13.y + _e11.z * _e13.z + _e11.w * _e13.w);
    return c_4_;
}

void main() {
    vec2 _e0 = test_fma();
    int _e1 = test_integer_dot_product();
    return;
}

