@group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 64>;
@group(0) @binding(1) var<uniform> b: array<array<mat2x2<f32>, 2>, 20>;
@group(0) @binding(2) var<storage, read_write> out: array<mat2x2<f32>, 4>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) i: u32) {
    let x = a;
    let y = b;
    out[0] = x[i];
    out[1] = y[i][1];
}
