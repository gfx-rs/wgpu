// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_2 {
    int inner[4];
};
struct type_9 {
    metal::int2 inner[4];
};
struct type_11 {
    metal::float2x2 inner[4];
};
struct type_13 {
    type_2 inner[4];
};
struct type_15 {
    type_13 inner[4];
};

void takes_ptr(
    thread int& p
) {
    return;
}

void takes_array_ptr(
    thread type_2& p_1
) {
    return;
}

void takes_vec_ptr(
    thread metal::int2& p_2
) {
    return;
}

void takes_mat_ptr(
    thread metal::float2x2& p_3
) {
    return;
}

void local_var(
    uint i
) {
    type_2 arr = type_2 {1, 2, 3, 4};
    takes_ptr(arr.inner[metal::min(unsigned(i), 3u)]);
    takes_array_ptr(arr);
    return;
}

void mat_vec_ptrs(
    thread type_9& pv,
    thread type_11& pm,
    uint i_1
) {
    takes_vec_ptr(pv.inner[metal::min(unsigned(i_1), 3u)]);
    takes_mat_ptr(pm.inner[metal::min(unsigned(i_1), 3u)]);
    return;
}

void argument(
    thread type_2& v,
    uint i_2
) {
    takes_ptr(v.inner[metal::min(unsigned(i_2), 3u)]);
    return;
}

void argument_nested_x2_(
    thread type_13& v_1,
    uint i_3,
    uint j
) {
    takes_ptr(v_1.inner[metal::min(unsigned(i_3), 3u)].inner[metal::min(unsigned(j), 3u)]);
    takes_ptr(v_1.inner[metal::min(unsigned(i_3), 3u)].inner[0]);
    takes_ptr(v_1.inner[0].inner[metal::min(unsigned(j), 3u)]);
    takes_array_ptr(v_1.inner[metal::min(unsigned(i_3), 3u)]);
    return;
}

void argument_nested_x3_(
    thread type_15& v_2,
    uint i_4,
    uint j_1
) {
    takes_ptr(v_2.inner[metal::min(unsigned(i_4), 3u)].inner[0].inner[metal::min(unsigned(j_1), 3u)]);
    takes_ptr(v_2.inner[metal::min(unsigned(i_4), 3u)].inner[metal::min(unsigned(j_1), 3u)].inner[0]);
    takes_ptr(v_2.inner[0].inner[metal::min(unsigned(i_4), 3u)].inner[metal::min(unsigned(j_1), 3u)]);
    return;
}

void index_from_self(
    thread type_2& v_3,
    uint i_5
) {
    int _e3 = v_3.inner[metal::min(unsigned(i_5), 3u)];
    takes_ptr(v_3.inner[metal::min(unsigned(_e3), 3u)]);
    return;
}

void local_var_from_arg(
    type_2 a,
    uint i_6
) {
    type_2 b = {};
    b = a;
    takes_ptr(b.inner[metal::min(unsigned(i_6), 3u)]);
    return;
}

void let_binding(
    thread type_2& a_1,
    uint i_7
) {
    takes_ptr(a_1.inner[metal::min(unsigned(i_7), 3u)]);
    takes_ptr(a_1.inner[0]);
    return;
}

kernel void main_(
) {
    type_9 vec_ = {};
    type_11 mat = {};
    type_2 arr1d = {};
    type_13 arr2d = {};
    type_15 arr3d = {};
    local_var(1u);
    mat_vec_ptrs(vec_, mat, 1u);
    argument(arr1d, 1u);
    argument_nested_x2_(arr2d, 1u, 2u);
    argument_nested_x3_(arr3d, 1u, 2u);
    index_from_self(arr1d, 1u);
    local_var_from_arg(type_2 {1, 2, 3, 4}, 5u);
    let_binding(arr1d, 1u);
    return;
}
