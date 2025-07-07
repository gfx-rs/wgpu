fn takes_ptr(p: ptr<function, i32>) {
    return;
}

fn takes_array_ptr(p_1: ptr<function, array<i32, 4>>) {
    return;
}

fn takes_vec_ptr(p_2: ptr<function, vec2<i32>>) {
    return;
}

fn takes_mat_ptr(p_3: ptr<function, mat2x2<f32>>) {
    return;
}

fn local_var(i: u32) {
    var arr: array<i32, 4> = array<i32, 4>(1i, 2i, 3i, 4i);

    takes_ptr((&arr[i]));
    takes_array_ptr((&arr));
    return;
}

fn mat_vec_ptrs(pv: ptr<function, array<vec2<i32>, 4>>, pm: ptr<function, array<mat2x2<f32>, 4>>, i_1: u32) {
    takes_vec_ptr((&(*pv)[i_1]));
    takes_mat_ptr((&(*pm)[i_1]));
    return;
}

fn argument(v: ptr<function, array<i32, 4>>, i_2: u32) {
    takes_ptr((&(*v)[i_2]));
    return;
}

fn argument_nested_x2_(v_1: ptr<function, array<array<i32, 4>, 4>>, i_3: u32, j: u32) {
    takes_ptr((&(*v_1)[i_3][j]));
    takes_ptr((&(*v_1)[i_3][0]));
    takes_ptr((&(*v_1)[0][j]));
    takes_array_ptr((&(*v_1)[i_3]));
    return;
}

fn argument_nested_x3_(v_2: ptr<function, array<array<array<i32, 4>, 4>, 4>>, i_4: u32, j_1: u32) {
    takes_ptr((&(*v_2)[i_4][0][j_1]));
    takes_ptr((&(*v_2)[i_4][j_1][0]));
    takes_ptr((&(*v_2)[0][i_4][j_1]));
    return;
}

fn index_from_self(v_3: ptr<function, array<i32, 4>>, i_5: u32) {
    let _e3 = (*v_3)[i_5];
    takes_ptr((&(*v_3)[_e3]));
    return;
}

fn local_var_from_arg(a: array<i32, 4>, i_6: u32) {
    var b: array<i32, 4>;

    b = a;
    takes_ptr((&b[i_6]));
    return;
}

fn let_binding(a_1: ptr<function, array<i32, 4>>, i_7: u32) {
    let p0_ = (&(*a_1)[i_7]);
    takes_ptr(p0_);
    let p1_ = (&(*a_1)[0]);
    takes_ptr(p1_);
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var vec: array<vec2<i32>, 4>;
    var mat: array<mat2x2<f32>, 4>;
    var arr1d: array<i32, 4>;
    var arr2d: array<array<i32, 4>, 4>;
    var arr3d: array<array<array<i32, 4>, 4>, 4>;

    local_var(1u);
    mat_vec_ptrs((&vec), (&mat), 1u);
    argument((&arr1d), 1u);
    argument_nested_x2_((&arr2d), 1u, 2u);
    argument_nested_x3_((&arr3d), 1u, 2u);
    index_from_self((&arr1d), 1u);
    local_var_from_arg(array<i32, 4>(1i, 2i, 3i, 4i), 5u);
    let_binding((&arr1d), 1u);
    return;
}
