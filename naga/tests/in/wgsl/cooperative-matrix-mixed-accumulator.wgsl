enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage, read_write> ab: array<f16>;
@group(0) @binding(1)
var<storage, read_write> accum: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main() {
    // loading f16 multiplicands from memory
    let a: coop_mat8x8<f16, A> = coopLoad<coop_mat8x8<f16, A>>(&ab[0]);
    let b: coop_mat8x8<f16, B> = coopLoad<coop_mat8x8<f16, B>>(&ab[0]);
    // loading f32 accumulator from memory
    var c: coop_mat8x8<f32, C> = coopLoad<coop_mat8x8<f32, C>>(&accum[0]);

    // actual mixed-accumulator multiply-add
    c = coopMultiplyAdd(a, b, c);
    // storing f32 accumulator into memory
    coopStore(c, &accum[0]);
}
