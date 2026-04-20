struct S {
  a : i32,
  b : mat2x2<f32>,
}

@group(0) @binding(0) var<uniform> u : S;

@compute @workgroup_size(1)
fn main() {
    let v = u;
}