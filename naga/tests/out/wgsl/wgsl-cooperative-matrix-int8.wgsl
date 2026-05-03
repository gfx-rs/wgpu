enable wgpu_cooperative_matrix;

var<private> a_i8_: coop_mat32x16<i8,A>;
var<private> b_i8_: coop_mat16x32<i8,B>;
@group(0) @binding(0) 
var<storage, read_write> ext_i8_: array<i8>;
var<private> a_u8_: coop_mat32x16<u8,A>;
var<private> b_u8_: coop_mat16x32<u8,B>;
@group(0) @binding(1) 
var<storage, read_write> ext_u8_: array<u8>;

@compute @workgroup_size(8, 8, 1) 
fn main_i8_() {
    var c: coop_mat16x16<i8,C>;
    var d: coop_mat16x16<i8,C>;

    c = coopLoad<coop_mat16x16<i8,C>>((&ext_i8_[4]), 16u);
    let _e6 = a_i8_;
    let _e8 = b_i8_;
    let _e9 = c;
    d = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d;
    coopStore(_e12, (&ext_i8_[0]), 16u);
    let _e16 = d;
    c = _e16;
    return;
}

@compute @workgroup_size(8, 8, 1) 
fn main_u8_() {
    var c_1: coop_mat16x16<u8,C>;
    var d_1: coop_mat16x16<u8,C>;

    c_1 = coopLoad<coop_mat16x16<u8,C>>((&ext_u8_[4]), 16u);
    let _e6 = a_u8_;
    let _e8 = b_u8_;
    let _e9 = c_1;
    d_1 = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d_1;
    coopStore(_e12, (&ext_u8_[0]), 16u);
    let _e16 = d_1;
    c_1 = _e16;
    return;
}
