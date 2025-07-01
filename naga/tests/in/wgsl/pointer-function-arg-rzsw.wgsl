fn takes_ptr(p: ptr<function, i32>) {}
fn takes_array_ptr(p: ptr<function, array<i32, 4>>) {}
fn takes_vec_ptr(p: ptr<function, vec2<i32>>) {}
fn takes_mat_ptr(p: ptr<function, mat2x2<f32>>) {}

fn local_var(i: u32) {
    var arr = array(1, 2, 3, 4);
    takes_ptr(&arr[i]);
    takes_array_ptr(&arr);

}

fn mat_vec_ptrs(
    pv: ptr<function, array<vec2<i32>, 4>>,
    pm: ptr<function, array<mat2x2<f32>, 4>>,
    i: u32,
) {
    takes_vec_ptr(&pv[i]);
    takes_mat_ptr(&pm[i]);
}

fn argument(v: ptr<function, array<i32, 4>>, i: u32) {
    takes_ptr(&v[i]);
}

fn argument_nested_x2(v: ptr<function, array<array<i32, 4>, 4>>, i: u32, j: u32) {
    takes_ptr(&v[i][j]);

    // Mixing compile and runtime bounds checks
    takes_ptr(&v[i][0]);
    takes_ptr(&v[0][j]);

    takes_array_ptr(&v[i]);
}

fn argument_nested_x3(v: ptr<function, array<array<array<i32, 4>, 4>, 4>>, i: u32, j: u32) {
    takes_ptr(&v[i][0][j]);
    takes_ptr(&v[i][j][0]);
    takes_ptr(&v[0][i][j]);
}

fn index_from_self(v: ptr<function, array<i32, 4>>, i: u32) {
    takes_ptr(&v[v[i]]);
}

fn local_var_from_arg(a: array<i32, 4>, i: u32) {
    var b = a;
    takes_ptr(&b[i]);
}

fn let_binding(a: ptr<function, array<i32, 4>>, i: u32) {
    let p0 = &a[i];
    takes_ptr(p0);

    let p1 = &a[0];
    takes_ptr(p1);
}

// Runtime-sized arrays can only appear in storage buffers, while (in the base
// language) pointers can only appear in function or private space, so there
// is no interaction to test.

@compute @workgroup_size(1)
fn main() {
    var vec: array<vec2<i32>, 4>;
    var mat: array<mat2x2<f32>, 4>;
    var arr1d: array<i32, 4>;
    var arr2d: array<array<i32, 4>, 4>;
    var arr3d: array<array<array<i32, 4>, 4>, 4>;
    local_var(1);
    mat_vec_ptrs(&vec, &mat, 1);
    argument(&arr1d, 1);
    argument_nested_x2(&arr2d, 1, 2);
    argument_nested_x3(&arr3d, 1, 2);
    index_from_self(&arr1d, 1);
    local_var_from_arg(array(1, 2, 3, 4), 5);
    let_binding(&arr1d, 1);
}
