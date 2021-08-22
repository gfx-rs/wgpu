// Tests for `naga::back::BoundsCheckPolicy::ReadZeroSkipWrite`.

[[block]]
struct Globals {
    a: array<f32, 10>;
    v: vec4<f32>;
    m: mat3x4<f32>;
};

[[group(0), binding(0)]] var<storage> globals: Globals;

fn index_array(i: i32) -> f32 {
   return globals.a[i];
}

fn index_vector(i: i32) -> f32 {
   return globals.v[i];
}

fn index_vector_by_value(v: vec4<f32>, i: i32) -> f32 {
   return v[i];
}

fn index_matrix(i: i32) -> vec4<f32> {
   return globals.m[i];
}

fn index_twice(i: i32, j: i32) -> f32 {
   return globals.m[i][j];
}

fn set_array(i: i32, v: f32) {
   globals.a[i] = v;
}

fn set_vector(i: i32, v: f32) {
   globals.v[i] = v;
}

fn set_matrix(i: i32, v: vec4<f32>) {
   globals.m[i] = v;
}

fn set_index_twice(i: i32, j: i32, v: f32) {
   globals.m[i][j] = v;
}
