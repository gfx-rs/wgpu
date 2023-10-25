#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;

const uint TWO = 2u;
const int THREE = 3;
const int FOUR = 4;
const int FOUR_ALIAS = 4;
const int TEST_CONSTANT_ADDITION = 8;
const int TEST_CONSTANT_ALIAS_ADDITION = 8;
const float PI = 3.141;
const float phi_sun = 6.282;
const vec4 DIV = vec4(0.44444445, 0.0, 0.0, 0.0);
const int TEXTURE_KIND_REGULAR = 0;
const int TEXTURE_KIND_WARP = 1;
const int TEXTURE_KIND_SKY = 2;


void swizzle_of_compose() {
    ivec4 out_ = ivec4(4, 3, 2, 1);
}

void index_of_compose() {
    int out_1 = 2;
}

void compose_three_deep() {
    int out_2 = 6;
}

void non_constant_initializers() {
    int w = 30;
    int x = 0;
    int y = 0;
    int z = 70;
    ivec4 out_3 = ivec4(0);
    int _e2 = w;
    x = _e2;
    int _e4 = x;
    y = _e4;
    int _e8 = w;
    int _e9 = x;
    int _e10 = y;
    int _e11 = z;
    out_3 = ivec4(_e8, _e9, _e10, _e11);
    return;
}

void splat_of_constant() {
    ivec4 out_4 = ivec4(-4, -4, -4, -4);
}

void compose_of_constant() {
    ivec4 out_5 = ivec4(-4, -4, -4, -4);
}

uint map_texture_kind(int texture_kind) {
    switch(texture_kind) {
        case 0: {
            return 10u;
        }
        case 1: {
            return 20u;
        }
        case 2: {
            return 30u;
        }
        default: {
            return 0u;
        }
    }
}

void main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    return;
}

