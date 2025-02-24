@group(0) @binding(0)
var<storage, read_write> in: u32;
@group(0) @binding(1)
var<storage, read_write> out: array<u32>;

struct T {
    @size(16)
    t: u32
}

@group(0) @binding(2)
var<uniform> in_data_uniform: array<T, 1>;

@group(0) @binding(3)
var<storage, read_write> in_data_storage_g0_b3: array<T, 1>;

@group(0) @binding(4)
var<storage, read_write> in_data_storage_g0_b4: array<T, 1>;

@group(1) @binding(0)
var<storage, read_write> in_data_storage_g1_b0: array<T, 1>;

@compute @workgroup_size(1)
fn main() {
    let i = in;
    out[0] = in_data_uniform[i].t;
    out[1] = in_data_storage_g0_b3[i].t;
    out[2] = in_data_storage_g0_b4[i].t;
    out[3] = in_data_storage_g1_b0[i].t;
}