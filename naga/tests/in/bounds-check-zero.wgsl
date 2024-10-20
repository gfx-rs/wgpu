// Tests for `naga::back::BoundsCheckPolicy::ReadZeroSkipWrite`.

struct Globals {
    a: array<f32, 10>,
    v: vec4<f32>,
    m: mat3x4<f32>,
    d: array<f32>,
}

@group(0) @binding(0) var<storage, read_write> globals: Globals;

fn index_array(i: i32) -> f32 {
   return globals.a[i];
}

fn index_dynamic_array(i: i32) -> f32 {
   return globals.d[i];
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

fn index_expensive(i: i32) -> f32 {
   return globals.a[i32(sin(f32(i) / 100.0) * 100.0)];
}

fn index_in_bounds() -> f32 {
   return globals.a[9] + globals.v[3] + globals.m[2][3];
}

fn set_array(i: i32, v: f32) {
   globals.a[i] = v;
}

fn set_dynamic_array(i: i32, v: f32) {
   globals.d[i] = v;
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

fn set_expensive(i: i32, v: f32) {
   globals.a[i32(sin(f32(i) / 100.0) * 100.0)] = v;
}

fn set_in_bounds(v: f32) {
   globals.a[9] = v;
   globals.v[3] = v;
   globals.m[2][3] = v;
}

fn index_dynamic_array_constant_index() -> f32 {
   return globals.d[1000];
}

fn set_dynamic_array_constant_index(v: f32) {
   globals.d[1000] = v;
}
