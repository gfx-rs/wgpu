enable wgpu_cooperative_matrix;

// type declarations with different roles
var<private> a: coop_mat8x8<f32, A>;
var<private> b: coop_mat8x8<f32, B>;
@group(0) @binding(0)
var<storage, read_write> ext: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main() {
    // loading from memory
    var c = coopLoad<coop_mat8x8<f32, C>>(&ext[4]);
    // actual multiply-add
    var d = coopMultiplyAdd(a, b, c);
    // storing into memory
    coopStore(d, &ext[0]);
    // operations on the type
    c = d;
}
