struct Inner { m1: mat2x2<f32>, m2: mat4x2<f32> }
struct Outer { a: vec4<f32>, i: Inner }
@group(0) @binding(0) var<uniform> u1: Inner;
@group(0) @binding(1) var<uniform> u2: Outer;
@group(0) @binding(2) var<storage, read_write> o: array<f32, 16>;
@compute @workgroup_size(1) fn main() {
  let k = u1.m1;
  let m = u2.i.m2;
  o[0] = m[0][0] + m[1][0] + m[2][0] + m[3][0] + k[0][0];
}
