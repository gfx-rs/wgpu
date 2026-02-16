// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

constant metal::float4 v_f32_one = metal::float4(1.0, 1.0, 1.0, 1.0);
constant metal::float4 v_f32_zero = metal::float4(0.0, 0.0, 0.0, 0.0);
constant metal::float4 v_f32_half = metal::float4(0.5, 0.5, 0.5, 0.5);
constant metal::int4 v_i32_one = metal::int4(1, 1, 1, 1);
constant bool b_false = false;
constant bool b_true = true;
constant bool short_circuit_1_invalid_rhs = false;
constant bool short_circuit_2_invalid_rhs = false;
constant bool short_circuit_3_ = true;
constant bool short_circuit_4_ = true;

metal::float4 builtins(
) {
    int s1_ = true ? 1 : 0;
    metal::float4 s2_ = true ? v_f32_one : v_f32_zero;
    metal::float4 s3_ = metal::float4(1.0, 1.0, 1.0, 1.0);
    metal::float4 m1_ = metal::mix(v_f32_zero, v_f32_one, v_f32_half);
    metal::float4 m2_ = metal::mix(v_f32_zero, v_f32_one, 0.1);
    float b1_ = as_type<float>(1);
    metal::float4 b2_ = as_type<metal::float4>(v_i32_one);
    metal::int4 v_i32_zero = metal::int4(0, 0, 0, 0);
    return ((((static_cast<metal::float4>(as_type<metal::int4>(as_type<metal::uint4>(metal::int4(s1_)) + as_type<metal::uint4>(v_i32_zero))) + s2_) + m1_) + m2_) + metal::float4(b1_)) + b2_;
}

metal::int4 naga_mod(metal::int4 lhs, metal::int4 rhs) {
    metal::int4 divisor = metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
    return lhs - (lhs / divisor) * divisor;
}

metal::float4 splat(
    float m,
    int n
) {
    metal::float2 a_2 = ((metal::float2(2.0) + metal::float2(m)) - metal::float2(4.0)) / metal::float2(8.0);
    metal::int4 b = naga_mod(metal::int4(n), metal::int4(2));
    return a_2.xyxy + static_cast<metal::float4>(b);
}

metal::float2 splat_assignment(
) {
    metal::float2 a = metal::float2(2.0);
    metal::float2 _e4 = a;
    a = _e4 + metal::float2(1.0);
    metal::float2 _e8 = a;
    a = _e8 - metal::float2(3.0);
    metal::float2 _e12 = a;
    a = _e12 / metal::float2(4.0);
    metal::float2 _e15 = a;
    return _e15;
}

metal::float3 bool_cast(
    metal::float3 x
) {
    metal::bool3 y = static_cast<metal::bool3>(x);
    return static_cast<metal::float3>(y);
}

bool p(
) {
    return true;
}

bool q(
) {
    return false;
}

bool r(
) {
    return true;
}

bool s(
) {
    return false;
}

void logical(
) {
    bool local = {};
    bool local_1 = {};
    bool local_2 = {};
    bool local_3 = {};
    bool local_4 = {};
    bool local_5 = {};
    bool local_6 = {};
    bool neg0_ = !(true);
    metal::bool2 neg1_ = !(metal::bool2(true));
    if (!(true)) {
        local = false;
    } else {
        local = true;
    }
    bool or_ = local;
    if (true) {
        local_1 = false;
    } else {
        local_1 = false;
    }
    bool and_ = local_1;
    bool bitwise_or0_ = true | false;
    metal::bool3 bitwise_or1_ = metal::bool3(true) | metal::bool3(false);
    bool bitwise_and0_ = true & false;
    metal::bool4 bitwise_and1_ = metal::bool4(true) & metal::bool4(false);
    if (!(false)) {
        local_2 = false;
    } else {
        local_2 = true;
    }
    bool _e27 = local_2;
    bool short_circuit_5_ = !(_e27);
    bool _e29 = p();
    if (!(_e29)) {
        bool _e33 = q();
        local_3 = _e33;
    } else {
        local_3 = true;
    }
    bool _e35 = local_3;
    if (_e35) {
        bool _e38 = r();
        if (!(_e38)) {
            bool _e42 = s();
            local_5 = _e42;
        } else {
            local_5 = true;
        }
        bool _e44 = local_5;
        local_4 = _e44;
    } else {
        local_4 = false;
    }
    bool short_circuit_6_ = local_4;
    if (false) {
        bool _e50 = q();
        local_6 = _e50;
    } else {
        local_6 = true;
    }
    bool short_circuit_7_ = local_6;
    return;
}

metal::int2 naga_neg(metal::int2 val) {
    return as_type<metal::int2>(-as_type<metal::uint2>(val));
}

int naga_div(int lhs, int rhs) {
    return lhs / metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

metal::int2 naga_div(metal::int2 lhs, metal::int2 rhs) {
    return lhs / metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
}

metal::uint3 naga_div(metal::uint3 lhs, metal::uint3 rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

int naga_mod(int lhs, int rhs) {
    int divisor = metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
    return lhs - (lhs / divisor) * divisor;
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

metal::int2 naga_mod(metal::int2 lhs, metal::int2 rhs) {
    metal::int2 divisor = metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
    return lhs - (lhs / divisor) * divisor;
}

metal::uint3 naga_mod(metal::uint3 lhs, metal::uint3 rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

metal::uint2 naga_div(metal::uint2 lhs, metal::uint2 rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

metal::uint2 naga_mod(metal::uint2 lhs, metal::uint2 rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

void arithmetic(
) {
    int prevent_const_eval = {};
    int wgpu_7437_ = {};
    float neg0_1 = -(1.0);
    metal::int2 neg1_1 = naga_neg(metal::int2(1));
    metal::float2 neg2_ = -(metal::float2(1.0));
    int add0_ = as_type<int>(as_type<uint>(2) + as_type<uint>(1));
    uint add1_ = 2u + 1u;
    float add2_ = 2.0 + 1.0;
    metal::int2 add3_ = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) + as_type<metal::uint2>(metal::int2(1)));
    metal::uint3 add4_ = metal::uint3(2u) + metal::uint3(1u);
    metal::float4 add5_ = metal::float4(2.0) + metal::float4(1.0);
    int sub0_ = as_type<int>(as_type<uint>(2) - as_type<uint>(1));
    uint sub1_ = 2u - 1u;
    float sub2_ = 2.0 - 1.0;
    metal::int2 sub3_ = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) - as_type<metal::uint2>(metal::int2(1)));
    metal::uint3 sub4_ = metal::uint3(2u) - metal::uint3(1u);
    metal::float4 sub5_ = metal::float4(2.0) - metal::float4(1.0);
    int mul0_ = as_type<int>(as_type<uint>(2) * as_type<uint>(1));
    uint mul1_ = 2u * 1u;
    float mul2_ = 2.0 * 1.0;
    metal::int2 mul3_ = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) * as_type<metal::uint2>(metal::int2(1)));
    metal::uint3 mul4_ = metal::uint3(2u) * metal::uint3(1u);
    metal::float4 mul5_ = metal::float4(2.0) * metal::float4(1.0);
    int div0_ = naga_div(2, 1);
    uint div1_ = naga_div(2u, 1u);
    float div2_ = 2.0 / 1.0;
    metal::int2 div3_ = naga_div(metal::int2(2), metal::int2(1));
    metal::uint3 div4_ = naga_div(metal::uint3(2u), metal::uint3(1u));
    metal::float4 div5_ = metal::float4(2.0) / metal::float4(1.0);
    int rem0_ = naga_mod(2, 1);
    uint rem1_ = naga_mod(2u, 1u);
    float rem2_ = metal::fmod(2.0, 1.0);
    metal::int2 rem3_ = naga_mod(metal::int2(2), metal::int2(1));
    metal::uint3 rem4_ = naga_mod(metal::uint3(2u), metal::uint3(1u));
    metal::float4 rem5_ = metal::fmod(metal::float4(2.0), metal::float4(1.0));
    {
        metal::int2 add0_1 = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) + as_type<metal::uint2>(metal::int2(1)));
        metal::int2 add1_1 = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) + as_type<metal::uint2>(metal::int2(1)));
        metal::uint2 add2_1 = metal::uint2(2u) + metal::uint2(1u);
        metal::uint2 add3_1 = metal::uint2(2u) + metal::uint2(1u);
        metal::float2 add4_1 = metal::float2(2.0) + metal::float2(1.0);
        metal::float2 add5_1 = metal::float2(2.0) + metal::float2(1.0);
        metal::int2 sub0_1 = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) - as_type<metal::uint2>(metal::int2(1)));
        metal::int2 sub1_1 = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) - as_type<metal::uint2>(metal::int2(1)));
        metal::uint2 sub2_1 = metal::uint2(2u) - metal::uint2(1u);
        metal::uint2 sub3_1 = metal::uint2(2u) - metal::uint2(1u);
        metal::float2 sub4_1 = metal::float2(2.0) - metal::float2(1.0);
        metal::float2 sub5_1 = metal::float2(2.0) - metal::float2(1.0);
        metal::int2 mul0_1 = as_type<metal::int2>(as_type<metal::uint2>(metal::int2(2)) * as_type<uint>(1));
        metal::int2 mul1_1 = as_type<metal::int2>(as_type<uint>(2) * as_type<metal::uint2>(metal::int2(1)));
        metal::uint2 mul2_1 = metal::uint2(2u) * 1u;
        metal::uint2 mul3_1 = 2u * metal::uint2(1u);
        metal::float2 mul4_1 = metal::float2(2.0) * 1.0;
        metal::float2 mul5_1 = 2.0 * metal::float2(1.0);
        metal::int2 div0_1 = naga_div(metal::int2(2), metal::int2(1));
        metal::int2 div1_1 = naga_div(metal::int2(2), metal::int2(1));
        metal::uint2 div2_1 = naga_div(metal::uint2(2u), metal::uint2(1u));
        metal::uint2 div3_1 = naga_div(metal::uint2(2u), metal::uint2(1u));
        metal::float2 div4_1 = metal::float2(2.0) / metal::float2(1.0);
        metal::float2 div5_1 = metal::float2(2.0) / metal::float2(1.0);
        metal::int2 rem0_1 = naga_mod(metal::int2(2), metal::int2(1));
        metal::int2 rem1_1 = naga_mod(metal::int2(2), metal::int2(1));
        metal::uint2 rem2_1 = naga_mod(metal::uint2(2u), metal::uint2(1u));
        metal::uint2 rem3_1 = naga_mod(metal::uint2(2u), metal::uint2(1u));
        metal::float2 rem4_1 = metal::fmod(metal::float2(2.0), metal::float2(1.0));
        metal::float2 rem5_1 = metal::fmod(metal::float2(2.0), metal::float2(1.0));
    }
    metal::float3x3 add = metal::float3x3 {} + metal::float3x3 {};
    metal::float3x3 sub = metal::float3x3 {} - metal::float3x3 {};
    metal::float3x3 mul_scalar0_ = metal::float3x3 {} * 1.0;
    metal::float3x3 mul_scalar1_ = 2.0 * metal::float3x3 {};
    metal::float3 mul_vector0_ = metal::float4x3 {} * metal::float4(1.0);
    metal::float4 mul_vector1_ = metal::float3(2.0) * metal::float4x3 {};
    metal::float3x3 mul = metal::float4x3 {} * metal::float3x4 {};
    int _e175 = prevent_const_eval;
    wgpu_7437_ = as_type<int>(as_type<uint>(_e175) + as_type<uint>((-2147483647 - 1)));
    return;
}

void bit(
) {
    int flip0_ = ~(1);
    uint flip1_ = ~(1u);
    metal::int2 flip2_ = ~(metal::int2(1));
    metal::uint3 flip3_ = ~(metal::uint3(1u));
    int or0_ = 2 | 1;
    uint or1_ = 2u | 1u;
    metal::int2 or2_ = metal::int2(2) | metal::int2(1);
    metal::uint3 or3_ = metal::uint3(2u) | metal::uint3(1u);
    int and0_ = 2 & 1;
    uint and1_ = 2u & 1u;
    metal::int2 and2_ = metal::int2(2) & metal::int2(1);
    metal::uint3 and3_ = metal::uint3(2u) & metal::uint3(1u);
    int xor0_ = 2 ^ 1;
    uint xor1_ = 2u ^ 1u;
    metal::int2 xor2_ = metal::int2(2) ^ metal::int2(1);
    metal::uint3 xor3_ = metal::uint3(2u) ^ metal::uint3(1u);
    int shl0_ = 2 << 1u;
    uint shl1_ = 2u << 1u;
    metal::int2 shl2_ = metal::int2(2) << metal::uint2(1u);
    metal::uint3 shl3_ = metal::uint3(2u) << metal::uint3(1u);
    int shr0_ = 2 >> 1u;
    uint shr1_ = 2u >> 1u;
    metal::int2 shr2_ = metal::int2(2) >> metal::uint2(1u);
    metal::uint3 shr3_ = metal::uint3(2u) >> metal::uint3(1u);
    return;
}

void comparison(
) {
    bool eq0_ = 2 == 1;
    bool eq1_ = 2u == 1u;
    bool eq2_ = 2.0 == 1.0;
    metal::bool2 eq3_ = metal::int2(2) == metal::int2(1);
    metal::bool3 eq4_ = metal::uint3(2u) == metal::uint3(1u);
    metal::bool4 eq5_ = metal::float4(2.0) == metal::float4(1.0);
    bool neq0_ = 2 != 1;
    bool neq1_ = 2u != 1u;
    bool neq2_ = 2.0 != 1.0;
    metal::bool2 neq3_ = metal::int2(2) != metal::int2(1);
    metal::bool3 neq4_ = metal::uint3(2u) != metal::uint3(1u);
    metal::bool4 neq5_ = metal::float4(2.0) != metal::float4(1.0);
    bool lt0_ = 2 < 1;
    bool lt1_ = 2u < 1u;
    bool lt2_ = 2.0 < 1.0;
    metal::bool2 lt3_ = metal::int2(2) < metal::int2(1);
    metal::bool3 lt4_ = metal::uint3(2u) < metal::uint3(1u);
    metal::bool4 lt5_ = metal::float4(2.0) < metal::float4(1.0);
    bool lte0_ = 2 <= 1;
    bool lte1_ = 2u <= 1u;
    bool lte2_ = 2.0 <= 1.0;
    metal::bool2 lte3_ = metal::int2(2) <= metal::int2(1);
    metal::bool3 lte4_ = metal::uint3(2u) <= metal::uint3(1u);
    metal::bool4 lte5_ = metal::float4(2.0) <= metal::float4(1.0);
    bool gt0_ = 2 > 1;
    bool gt1_ = 2u > 1u;
    bool gt2_ = 2.0 > 1.0;
    metal::bool2 gt3_ = metal::int2(2) > metal::int2(1);
    metal::bool3 gt4_ = metal::uint3(2u) > metal::uint3(1u);
    metal::bool4 gt5_ = metal::float4(2.0) > metal::float4(1.0);
    bool gte0_ = 2 >= 1;
    bool gte1_ = 2u >= 1u;
    bool gte2_ = 2.0 >= 1.0;
    metal::bool2 gte3_ = metal::int2(2) >= metal::int2(1);
    metal::bool3 gte4_ = metal::uint3(2u) >= metal::uint3(1u);
    metal::bool4 gte5_ = metal::float4(2.0) >= metal::float4(1.0);
    return;
}

void assignment(
) {
    int a_1 = {};
    metal::int3 vec0_ = metal::int3 {};
    a_1 = 1;
    int _e5 = a_1;
    a_1 = as_type<int>(as_type<uint>(_e5) + as_type<uint>(1));
    int _e7 = a_1;
    a_1 = as_type<int>(as_type<uint>(_e7) - as_type<uint>(1));
    int _e9 = a_1;
    int _e10 = a_1;
    a_1 = as_type<int>(as_type<uint>(_e10) * as_type<uint>(_e9));
    int _e12 = a_1;
    int _e13 = a_1;
    a_1 = naga_div(_e13, _e12);
    int _e15 = a_1;
    a_1 = naga_mod(_e15, 1);
    int _e17 = a_1;
    a_1 = _e17 & 0;
    int _e19 = a_1;
    a_1 = _e19 | 0;
    int _e21 = a_1;
    a_1 = _e21 ^ 0;
    int _e23 = a_1;
    a_1 = _e23 << 2u;
    int _e25 = a_1;
    a_1 = _e25 >> 1u;
    int _e28 = a_1;
    a_1 = as_type<int>(as_type<uint>(_e28) + as_type<uint>(1));
    int _e31 = a_1;
    a_1 = as_type<int>(as_type<uint>(_e31) - as_type<uint>(1));
    int _e37 = vec0_[1];
    vec0_[1] = as_type<int>(as_type<uint>(_e37) + as_type<uint>(1));
    int _e41 = vec0_[1];
    vec0_[1] = as_type<int>(as_type<uint>(_e41) - as_type<uint>(1));
    return;
}

int naga_neg(int val) {
    return as_type<int>(-as_type<uint>(val));
}

void negation_avoids_prefix_decrement(
) {
    int i0_ = naga_neg(1);
    int i1_ = naga_neg(naga_neg(1));
    int i2_ = naga_neg(naga_neg(1));
    int i3_ = naga_neg(naga_neg(1));
    int i4_ = naga_neg(naga_neg(naga_neg(1)));
    int i5_ = naga_neg(naga_neg(naga_neg(naga_neg(1))));
    int i6_ = naga_neg(naga_neg(naga_neg(naga_neg(naga_neg(1)))));
    int i7_ = naga_neg(naga_neg(naga_neg(naga_neg(naga_neg(1)))));
    float f0_ = -(1.0);
    float f1_ = -(-(1.0));
    float f2_ = -(-(1.0));
    float f3_ = -(-(1.0));
    float f4_ = -(-(-(1.0)));
    float f5_ = -(-(-(-(1.0))));
    float f6_ = -(-(-(-(-(1.0)))));
    float f7_ = -(-(-(-(-(1.0)))));
    return;
}

struct main_Input {
};
kernel void main_(
  metal::uint3 id [[threadgroup_position_in_grid]]
) {
    metal::float4 _e1 = builtins();
    metal::float4 _e6 = splat(static_cast<float>(id.x), static_cast<int>(id.y));
    metal::float2 _e7 = splat_assignment();
    metal::float3 _e12 = bool_cast(metal::float3(1.0, 1.0, 1.0));
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    negation_avoids_prefix_decrement();
    return;
}
