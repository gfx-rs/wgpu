enable wgpu_cooperative_matrix;

var<private> a: coop_mat8x8<f32,A>;
var<private> b: coop_mat8x8<f32,B>;
@group(0) @binding(0) 
var<storage, read_write> ext: array<f32>;

@compute @workgroup_size(8, 8, 1) 
fn main() {
    var c: coop_mat8x8<f32,C>;
    var d: coop_mat8x8<f32,C>;

    c = coopLoad<coop_mat8x8<f32,C>>((&ext[4]), 8u);
    let _e6 = a;
    let _e8 = b;
    let _e9 = c;
    d = coopMultiplyAdd(_e6, _e8, _e9);
    let _e12 = d;
    coopStore(_e12, (&ext[0]), 8u);
    let _e16 = d;
    c = _e16;
    return;
}
