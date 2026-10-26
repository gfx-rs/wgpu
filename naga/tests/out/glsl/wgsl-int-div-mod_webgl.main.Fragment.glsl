#version 300 es

precision highp float;
precision highp int;

struct Inputs {
    ivec4 a;
    ivec4 b;
    uvec4 ua;
    uvec4 ub;
};
layout(std140) uniform Inputs_block_0Fragment { Inputs _group_0_binding_0_fs; };

layout(location = 0) out ivec4 _fs2p_location0;

int naga_div(int lhs, int rhs) {
    int divisor = (rhs == 0 || (lhs == (-2147483647 - 1) && rhs == -1)) ? 1 : rhs;
    return lhs / divisor;
}

int naga_mod(int lhs, int rhs) {
    int divisor = (rhs == 0 || (lhs == (-2147483647 - 1) && rhs == -1)) ? 1 : rhs;
    return lhs - divisor * (lhs / divisor);
}

uint naga_div(uint lhs, uint rhs) {
    uint divisor = rhs == 0u ? 1u : rhs;
    return lhs / divisor;
}

uint naga_mod(uint lhs, uint rhs) {
    uint divisor = rhs == 0u ? 1u : rhs;
    return lhs % divisor;
}

ivec2 naga_div(ivec2 lhs, ivec2 rhs) {
    ivec2 divisor = ivec2(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y);
    return lhs / divisor;
}

ivec2 naga_mod(ivec2 lhs, ivec2 rhs) {
    ivec2 divisor = ivec2(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y);
    return lhs - divisor * (lhs / divisor);
}

ivec3 naga_mod(ivec3 lhs, ivec3 rhs) {
    ivec3 divisor = ivec3(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y,
        (rhs.z == 0 || (lhs.z == (-2147483647 - 1) && rhs.z == -1)) ? 1 : rhs.z);
    return lhs - divisor * (lhs / divisor);
}

ivec3 naga_div(ivec3 lhs, ivec3 rhs) {
    ivec3 divisor = ivec3(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y,
        (rhs.z == 0 || (lhs.z == (-2147483647 - 1) && rhs.z == -1)) ? 1 : rhs.z);
    return lhs / divisor;
}

ivec4 naga_div(ivec4 lhs, ivec4 rhs) {
    ivec4 divisor = ivec4(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y,
        (rhs.z == 0 || (lhs.z == (-2147483647 - 1) && rhs.z == -1)) ? 1 : rhs.z,
        (rhs.w == 0 || (lhs.w == (-2147483647 - 1) && rhs.w == -1)) ? 1 : rhs.w);
    return lhs / divisor;
}

ivec4 naga_mod(ivec4 lhs, ivec4 rhs) {
    ivec4 divisor = ivec4(
        (rhs.x == 0 || (lhs.x == (-2147483647 - 1) && rhs.x == -1)) ? 1 : rhs.x,
        (rhs.y == 0 || (lhs.y == (-2147483647 - 1) && rhs.y == -1)) ? 1 : rhs.y,
        (rhs.z == 0 || (lhs.z == (-2147483647 - 1) && rhs.z == -1)) ? 1 : rhs.z,
        (rhs.w == 0 || (lhs.w == (-2147483647 - 1) && rhs.w == -1)) ? 1 : rhs.w);
    return lhs - divisor * (lhs / divisor);
}

uvec2 naga_div(uvec2 lhs, uvec2 rhs) {
    uvec2 divisor = uvec2(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y);
    return lhs / divisor;
}

uvec2 naga_mod(uvec2 lhs, uvec2 rhs) {
    uvec2 divisor = uvec2(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y);
    return lhs % divisor;
}

uvec3 naga_mod(uvec3 lhs, uvec3 rhs) {
    uvec3 divisor = uvec3(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y,
        rhs.z == 0u ? 1u : rhs.z);
    return lhs % divisor;
}

uvec3 naga_div(uvec3 lhs, uvec3 rhs) {
    uvec3 divisor = uvec3(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y,
        rhs.z == 0u ? 1u : rhs.z);
    return lhs / divisor;
}

uvec4 naga_div(uvec4 lhs, uvec4 rhs) {
    uvec4 divisor = uvec4(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y,
        rhs.z == 0u ? 1u : rhs.z,
        rhs.w == 0u ? 1u : rhs.w);
    return lhs / divisor;
}

uvec4 naga_mod(uvec4 lhs, uvec4 rhs) {
    uvec4 divisor = uvec4(
        rhs.x == 0u ? 1u : rhs.x,
        rhs.y == 0u ? 1u : rhs.y,
        rhs.z == 0u ? 1u : rhs.z,
        rhs.w == 0u ? 1u : rhs.w);
    return lhs % divisor;
}

int signed_scalar(int a, int b) {
    int x = 0;
    x = naga_div(a, b);
    int _e4 = x;
    x = (_e4 + naga_mod(a, b));
    int _e7 = x;
    x = naga_div(_e7, b);
    int _e9 = x;
    x = naga_mod(_e9, b);
    int _e11 = x;
    return _e11;
}

uint unsigned_scalar(uint a_1, uint b_1) {
    uint x_1 = 0u;
    x_1 = naga_div(a_1, b_1);
    uint _e4 = x_1;
    x_1 = (_e4 + naga_mod(a_1, b_1));
    uint _e7 = x_1;
    x_1 = naga_div(_e7, b_1);
    uint _e9 = x_1;
    x_1 = naga_mod(_e9, b_1);
    uint _e11 = x_1;
    return _e11;
}

ivec4 signed_vectors(ivec4 a_2, ivec4 b_2) {
    ivec2 x2_ = ivec2(0);
    ivec3 x3_ = ivec3(0);
    ivec4 x4_ = ivec4(0);
    x2_ = naga_div(a_2.xy, b_2.xy);
    ivec2 _e6 = x2_;
    x2_ = naga_mod(_e6, b_2.zw);
    x3_ = naga_mod(a_2.xyz, b_2.xyz);
    ivec3 _e13 = x3_;
    x3_ = naga_div(_e13, b_2.yzw);
    x4_ = (naga_div(a_2, b_2) + naga_mod(a_2, b_2));
    ivec4 _e20 = x4_;
    x4_ = (_e20 + naga_mod(naga_div(a_2, ivec4(b_2.x)), ivec4(b_2.y)));
    ivec4 _e28 = x4_;
    x4_ = (_e28 + (naga_div(ivec4(a_2.x), b_2) + naga_mod(ivec4(a_2.y), b_2)));
    ivec4 _e37 = x4_;
    ivec2 _e38 = x2_;
    ivec3 _e39 = x3_;
    return (_e37 + ivec4(_e38, _e39.xy));
}

uvec4 unsigned_vectors(uvec4 a_3, uvec4 b_3) {
    uvec2 x2_1 = uvec2(0u);
    uvec3 x3_1 = uvec3(0u);
    uvec4 x4_1 = uvec4(0u);
    x2_1 = naga_div(a_3.xy, b_3.xy);
    uvec2 _e6 = x2_1;
    x2_1 = naga_mod(_e6, b_3.zw);
    x3_1 = naga_mod(a_3.xyz, b_3.xyz);
    uvec3 _e13 = x3_1;
    x3_1 = naga_div(_e13, b_3.yzw);
    x4_1 = (naga_div(a_3, b_3) + naga_mod(a_3, b_3));
    uvec4 _e20 = x4_1;
    x4_1 = (_e20 + naga_mod(naga_div(a_3, uvec4(b_3.x)), uvec4(b_3.y)));
    uvec4 _e28 = x4_1;
    x4_1 = (_e28 + (naga_div(uvec4(a_3.x), b_3) + naga_mod(uvec4(a_3.y), b_3)));
    uvec4 _e37 = x4_1;
    uvec2 _e38 = x2_1;
    uvec3 _e39 = x3_1;
    return (_e37 + uvec4(_e38, _e39.xy));
}

void main() {
    int _e3 = _group_0_binding_0_fs.a.x;
    int _e7 = _group_0_binding_0_fs.b.x;
    int _e8 = signed_scalar(_e3, _e7);
    uint _e12 = _group_0_binding_0_fs.ua.x;
    uint _e16 = _group_0_binding_0_fs.ub.x;
    uint _e17 = unsigned_scalar(_e12, _e16);
    ivec4 _e20 = _group_0_binding_0_fs.a;
    ivec4 _e23 = _group_0_binding_0_fs.b;
    ivec4 _e24 = signed_vectors(_e20, _e23);
    uvec4 _e27 = _group_0_binding_0_fs.ua;
    uvec4 _e30 = _group_0_binding_0_fs.ub;
    uvec4 _e31 = unsigned_vectors(_e27, _e30);
    _fs2p_location0 = ((_e24 + ivec4(_e31)) + ivec4(_e8, int(_e17), 0, 0));
    return;
}

