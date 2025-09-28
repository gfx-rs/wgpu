var<private> a: coop_mat8x8<f32, A>;
var<private> b: coop_mat8x8<f32, B>;
@group(0) @binding(0)
var<storage, read_write> ext: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main() {
    var c = coopLoad(&ext[4]);
    var d = coopMultiplyAdd(a, b, c);
    coopStore(d, &ext[0]);
    c = d;
}
