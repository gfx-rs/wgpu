#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


uint naga_count_one_bits(uint e) {
    int r = bitCount(e);
    return uint(r);
}

uvec2 naga_count_one_bits(uvec2 e) {
    return uvec2(naga_count_one_bits(e.x), naga_count_one_bits(e.y));
}

int naga_count_one_bits(int e) {
    int r = bitCount(e);
    return r;
}

uint naga_count_trailing_zeros(uint e) {
    int r = findLSB(e);
    return min(uint(r), 32u);
}

int naga_count_trailing_zeros(int e) {
    int r = findLSB(e);
    return int(min(uint(r), 32u));
}

uint naga_first_trailing_bit(uint e) {
    int r = findLSB(e);
    return uint(r);
}

int naga_first_trailing_bit(int e) {
    int r = findLSB(e);
    return r;
}

uint naga_count_leading_zeros(uint e) {
    int r = findMSB(e);
    return uint(31 - r);
}

int naga_count_leading_zeros(int e) {
    int r = findMSB(e);
    return e < 0 ? 0 : 31 - r;
}

uint naga_first_leading_bit(uint e) {
    int r = findMSB(e);
    return uint(r);
}

int naga_first_leading_bit(int e) {
    int r = findMSB(e);
    return r;
}

void main() {
    uint u = 3735928559u;
    int i = -2;
    float f = 0.0;
    uvec2 u2_ = uvec2(3735928559u, 0u);
    uint _e10 = u;
    u = ~(naga_count_one_bits(_e10));
    uint _e13 = u;
    u = (naga_count_one_bits(_e13) * 65537u);
    uint _e17 = u;
    u = ~(naga_count_trailing_zeros(_e17));
    uint _e20 = u;
    u = ~(naga_first_trailing_bit(_e20));
    uint _e23 = u;
    u = ~(naga_count_leading_zeros(_e23));
    uint _e26 = u;
    u = ~(naga_first_leading_bit(_e26));
    int _e29 = i;
    i = (naga_count_one_bits(_e29) - 100000);
    int _e33 = i;
    i = ~(naga_count_trailing_zeros(_e33));
    int _e36 = i;
    i = ~(naga_first_trailing_bit(_e36));
    int _e39 = i;
    i = ~(naga_first_leading_bit(_e39));
    int _e42 = i;
    i = ~(naga_count_leading_zeros(_e42));
    uint _e45 = u;
    f = float(naga_first_trailing_bit(_e45));
    uint _e48 = u;
    f = float(naga_first_leading_bit(_e48));
    uvec2 _e51 = u2_;
    u2_ = (~(naga_count_one_bits(_e51)) * uvec2(65537u));
    return;
}

