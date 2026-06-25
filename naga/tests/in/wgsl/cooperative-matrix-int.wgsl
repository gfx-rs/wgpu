enable wgpu_cooperative_matrix;

// i32 cooperative matrix — square 16x16 (most portable integer config)
var<private> a_i32: coop_mat16x16<i32, A>;
var<private> b_i32: coop_mat16x16<i32, B>;
@group(0) @binding(0)
var<storage, read_write> ext_i32: array<i32>;

@compute @workgroup_size(8, 8, 1)
fn main_i32() {
    var c = coopLoad<coop_mat16x16<i32, C>>(&ext_i32[4]);
    var d = coopMultiplyAdd(a_i32, b_i32, c);
    coopStore(d, &ext_i32[0]);
    c = d;
}

// u32 cooperative matrix — square 16x16
var<private> a_u32: coop_mat16x16<u32, A>;
var<private> b_u32: coop_mat16x16<u32, B>;
@group(0) @binding(1)
var<storage, read_write> ext_u32: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main_u32() {
    var c = coopLoad<coop_mat16x16<u32, C>>(&ext_u32[4]);
    var d = coopMultiplyAdd(a_u32, b_u32, c);
    coopStore(d, &ext_u32[0]);
    c = d;
}
