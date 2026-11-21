struct Aligned {
    @align(16) m: mat3x2<f32>,
}

@group(0) @binding(0) var<uniform> u_m: mat3x2<f32>;
@group(0) @binding(1) var<uniform> u_arr: array<Aligned, 2>;
@group(0) @binding(2) var<uniform> u_v: array<vec4<f32>, 2>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>, 4>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) i: u32) {
    out[0] = u_m[i];
    out[1] = u_arr[i].m[1];
    out[2] = u_arr[i].m[i];
    out[3] = u_v[i].xy;
}
