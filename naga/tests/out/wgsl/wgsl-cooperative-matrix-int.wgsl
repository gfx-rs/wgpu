enable wgpu_cooperative_matrix;

var<private> a_i32_: coop_mat16x16<i32,A>;
var<private> b_i32_: coop_mat16x16<i32,B>;
@group(0) @binding(0) 
var<storage, read_write> ext_i32_: array<i32>;
var<private> a_u32_: coop_mat16x16<u32,A>;
var<private> b_u32_: coop_mat16x16<u32,B>;
@group(0) @binding(1) 
var<storage, read_write> ext_u32_: array<u32>;

@compute @workgroup_size(8, 8, 1) 
fn main_i32_() {
    var c: coop_mat16x16<i32,C>;
    var d: coop_mat16x16<i32,C>;

    c = coopLoad<coop_mat16x16<i32,C>>((&ext_i32_[4]), 16u);
    let _e6 = a_i32_;
    let _e8 = b_i32_;
    let _e9 = c;
    d = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d;
    coopStore(_e12, (&ext_i32_[0]), 16u);
    let _e16 = d;
    c = _e16;
    return;
}

@compute @workgroup_size(8, 8, 1) 
fn main_u32_() {
    var c_1: coop_mat16x16<u32,C>;
    var d_1: coop_mat16x16<u32,C>;

    c_1 = coopLoad<coop_mat16x16<u32,C>>((&ext_u32_[4]), 16u);
    let _e6 = a_u32_;
    let _e8 = b_u32_;
    let _e9 = c_1;
    d_1 = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d_1;
    coopStore(_e12, (&ext_u32_[0]), 16u);
    let _e16 = d_1;
    c_1 = _e16;
    return;
}
