var<private> a: coop_mat8x8<f32,A>;
var<private> b: coop_mat8x8<f32,B>;
@group(0) @binding(0) 
var<storage, read_write> ext: array<f32>;

@compute @workgroup_size(8, 8, 1) 
fn main() {
    var c: coop_mat8x8<f32,C> = coop_mat8x8<f32,C>();
    var d: coop_mat8x8<f32,C>;

    let _e2 = c;
    coopLoad(_e2, (&ext[4]), 8u);
    let _e7 = a;
    let _e9 = b;
    let _e10 = c;
    d = coopMultiplyAdd(_e7, _e9, _e10);
    let _e13 = d;
    coopStore(_e13, (&ext[0]), 8u);
    let _e17 = d;
    c = _e17;
    return;
}
