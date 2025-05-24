// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};

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
    int oob = {};
    takes_ptr(uint(i) < 4 ? arr.inner[i] : oob);
    takes_array_ptr(arr);
    return;
}

void mat_vec_ptrs(
    thread type_9& pv,
    thread type_11& pm,
    uint i_1
) {
    metal::int2 oob_1 = {};
    metal::float2x2 oob_2 = {};
    takes_vec_ptr(uint(i_1) < 4 ? pv.inner[i_1] : oob_1);
    takes_mat_ptr(uint(i_1) < 4 ? pm.inner[i_1] : oob_2);
    return;
}

void argument(
    thread type_2& v,
    uint i_2
) {
    int oob_3 = {};
    takes_ptr(uint(i_2) < 4 ? v.inner[i_2] : oob_3);
    return;
}

void argument_nested_x2_(
    thread type_13& v_1,
    uint i_3,
    uint j
) {
    int oob_4 = {};
    type_2 oob_5 = {};
    takes_ptr(uint(j) < 4 && uint(i_3) < 4 ? v_1.inner[i_3].inner[j] : oob_4);
    takes_ptr(uint(i_3) < 4 ? v_1.inner[i_3].inner[0] : oob_4);
    takes_ptr(uint(j) < 4 ? v_1.inner[0].inner[j] : oob_4);
    takes_array_ptr(uint(i_3) < 4 ? v_1.inner[i_3] : oob_5);
    return;
}

void argument_nested_x3_(
    thread type_15& v_2,
    uint i_4,
    uint j_1
) {
    int oob_6 = {};
    takes_ptr(uint(j_1) < 4 && uint(i_4) < 4 ? v_2.inner[i_4].inner[0].inner[j_1] : oob_6);
    takes_ptr(uint(j_1) < 4 && uint(i_4) < 4 ? v_2.inner[i_4].inner[j_1].inner[0] : oob_6);
    takes_ptr(uint(j_1) < 4 && uint(i_4) < 4 ? v_2.inner[0].inner[i_4].inner[j_1] : oob_6);
    return;
}

void index_from_self(
    thread type_2& v_3,
    uint i_5
) {
    int oob_7 = {};
    int _e3 = uint(i_5) < 4 ? v_3.inner[i_5] : DefaultConstructible();
    takes_ptr(uint(_e3) < 4 ? v_3.inner[_e3] : oob_7);
    return;
}

void local_var_from_arg(
    type_2 a,
    uint i_6
) {
    type_2 b = {};
    int oob_8 = {};
    b = a;
    takes_ptr(uint(i_6) < 4 ? b.inner[i_6] : oob_8);
    return;
}

void let_binding(
    thread type_2& a_1,
    uint i_7
) {
    int oob_9 = {};
    takes_ptr(uint(i_7) < 4 ? a_1.inner[i_7] : oob_9);
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
