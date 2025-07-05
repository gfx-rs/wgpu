#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void takes_ptr(inout int p) {
    return;
}

void takes_array_ptr(inout int p_1[4]) {
    return;
}

void takes_vec_ptr(inout ivec2 p_2) {
    return;
}

void takes_mat_ptr(inout mat2x2 p_3) {
    return;
}

void local_var(uint i) {
    int arr[4] = int[4](1, 2, 3, 4);
    takes_ptr(arr[i]);
    takes_array_ptr(arr);
    return;
}

void mat_vec_ptrs(inout ivec2 pv[4], inout mat2x2 pm[4], uint i_1) {
    takes_vec_ptr(pv[i_1]);
    takes_mat_ptr(pm[i_1]);
    return;
}

void argument(inout int v[4], uint i_2) {
    takes_ptr(v[i_2]);
    return;
}

void argument_nested_x2_(inout int v_1[4][4], uint i_3, uint j) {
    takes_ptr(v_1[i_3][j]);
    takes_ptr(v_1[i_3][0]);
    takes_ptr(v_1[0][j]);
    takes_array_ptr(v_1[i_3]);
    return;
}

void argument_nested_x3_(inout int v_2[4][4][4], uint i_4, uint j_1) {
    takes_ptr(v_2[i_4][0][j_1]);
    takes_ptr(v_2[i_4][j_1][0]);
    takes_ptr(v_2[0][i_4][j_1]);
    return;
}

void index_from_self(inout int v_3[4], uint i_5) {
    int _e3 = v_3[i_5];
    takes_ptr(v_3[_e3]);
    return;
}

void local_var_from_arg(int a[4], uint i_6) {
    int b[4] = int[4](0, 0, 0, 0);
    b = a;
    takes_ptr(b[i_6]);
    return;
}

void let_binding(inout int a_1[4], uint i_7) {
    takes_ptr(a_1[i_7]);
    takes_ptr(a_1[0]);
    return;
}

void main() {
    ivec2 vec[4] = ivec2[4](ivec2(0), ivec2(0), ivec2(0), ivec2(0));
    mat2x2 mat[4] = mat2x2[4](mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0));
    int arr1d[4] = int[4](0, 0, 0, 0);
    int arr2d[4][4] = int[4][4](int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0));
    int arr3d[4][4][4] = int[4][4][4](int[4][4](int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0)), int[4][4](int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0)), int[4][4](int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0)), int[4][4](int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0), int[4](0, 0, 0, 0)));
    local_var(1u);
    mat_vec_ptrs(vec, mat, 1u);
    argument(arr1d, 1u);
    argument_nested_x2_(arr2d, 1u, 2u);
    argument_nested_x3_(arr3d, 1u, 2u);
    index_from_self(arr1d, 1u);
    local_var_from_arg(int[4](1, 2, 3, 4), 5u);
    let_binding(arr1d, 1u);
    return;
}

