struct S {
    a: f32,
    m: mat3x2<f32>,
}

struct Aligned {
    @align(16) m: mat3x2<f32>,
}

@group(0) @binding(0) var<uniform> u_m: mat3x2<f32>;
@group(0) @binding(1) var<uniform> u_s: S;
@group(0) @binding(2) var<uniform> u_arr: array<Aligned, 2>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>, 8>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) i: u32) {
    out[0] = u_m[i];
    out[1] = u_s.m[i];
    out[2] = vec2(u_m[i][i], u_s.m[i][i]);
    out[3] = u_arr[i].m[i];
    out[4] = vec2(u_arr[i].m[i][i], 0.0);
}
