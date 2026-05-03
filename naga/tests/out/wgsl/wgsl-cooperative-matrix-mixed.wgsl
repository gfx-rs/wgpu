enable wgpu_cooperative_matrix;

var<private> a_f16_: coop_mat32x16<f16,A>;
var<private> b_f16_: coop_mat16x32<f16,B>;
@group(0) @binding(0) 
var<storage, read_write> ext_f32_: array<f32>;
var<private> a_i8_: coop_mat32x16<i8,A>;
var<private> b_i8_: coop_mat16x32<i8,B>;
@group(0) @binding(1) 
var<storage, read_write> ext_i32_: array<i32>;
var<private> a_u8_: coop_mat32x16<u8,A>;
var<private> b_u8_: coop_mat16x32<u8,B>;
@group(0) @binding(2) 
var<storage, read_write> ext_u32_: array<u32>;

@compute @workgroup_size(8, 8, 1) 
fn main_f16_f32_() {
    var c: coop_mat16x16<f32,C>;
    var d: coop_mat16x16<f32,C>;

    c = coopLoad<coop_mat16x16<f32,C>>((&ext_f32_[4]), 16u);
    let _e6 = a_f16_;
    let _e8 = b_f16_;
    let _e9 = c;
    d = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d;
    coopStore(_e12, (&ext_f32_[0]), 16u);
    let _e16 = d;
    c = _e16;
    return;
}

@compute @workgroup_size(8, 8, 1) 
fn main_i8_i32_() {
    var c_1: coop_mat16x16<i32,C>;
    var d_1: coop_mat16x16<i32,C>;

    c_1 = coopLoad<coop_mat16x16<i32,C>>((&ext_i32_[4]), 16u);
    let _e6 = a_i8_;
    let _e8 = b_i8_;
    let _e9 = c_1;
    d_1 = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d_1;
    coopStore(_e12, (&ext_i32_[0]), 16u);
    let _e16 = d_1;
    c_1 = _e16;
    return;
}

@compute @workgroup_size(8, 8, 1) 
fn main_u8_u32_() {
    var c_2: coop_mat16x16<u32,C>;
    var d_2: coop_mat16x16<u32,C>;

    c_2 = coopLoad<coop_mat16x16<u32,C>>((&ext_u32_[4]), 16u);
    let _e6 = a_u8_;
    let _e8 = b_u8_;
    let _e9 = c_2;
    d_2 = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d_2;
    coopStore(_e12, (&ext_u32_[0]), 16u);
    let _e16 = d_2;
    c_2 = _e16;
    return;
}
