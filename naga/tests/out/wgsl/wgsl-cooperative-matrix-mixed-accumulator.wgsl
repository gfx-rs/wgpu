enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) 
var<storage, read_write> ab: array<f16>;
@group(0) @binding(1) 
var<storage, read_write> accum: array<f32>;

@compute @workgroup_size(8, 8, 1) 
fn main() {
    var c: coop_mat8x8<f32,C>;

    let a = coopLoad<coop_mat8x8<f16,A>>((&ab[0]), 8u);
    let b = coopLoad<coop_mat8x8<f16,B>>((&ab[0]), 8u);
    c = coopLoad<coop_mat8x8<f32,C>>((&accum[0]), 8u);
    let _e13 = c;
    c = coopMultiplyAdd(a, b, _e13);
    let _e15 = c;
    coopStore(_e15, (&accum[0]), 8u);
    return;
}
