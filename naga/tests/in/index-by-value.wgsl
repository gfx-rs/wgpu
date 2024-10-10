fn index_arg_array(a: array<i32, 5>, i: i32) -> i32 {
    return a[i];
}

fn index_let_array(i: i32, j: i32) -> i32 {
  let a = array<array<i32, 2>, 2>(array(1, 2), array(3, 4));
  return a[i][j];
}

fn index_let_matrix(i: i32, j: i32) -> f32 {
  let a = mat2x2<f32>(1, 2, 3, 4);
  return a[i][j];
}
