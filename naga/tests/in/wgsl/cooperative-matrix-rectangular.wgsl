enable f16;
enable wgpu_cooperative_matrix;

// `coop_mat{columns}x{rows}`. Both entry points use K=16, f16 multiplicands,
// and an f32 accumulator. `tile_16x8` is M=8, N=16. `tile_8x16` is M=16, N=8.
@group(0) @binding(0)
var<storage, read_write> ab: array<f16>;
@group(0) @binding(1)
var<storage, read_write> accum: array<f32>;

@compute @workgroup_size(32, 1, 1)
fn tile_16x8() {
    let a = coopLoad<coop_mat16x8<f16, A>>(&ab[0]);
    let b = coopLoad<coop_mat16x16<f16, B>>(&ab[0]);
    var c = coopLoad<coop_mat16x8<f32, C>>(&accum[0]);
    c = coopMultiplyAdd(a, b, c);
    coopStore(c, &accum[0]);
}

@compute @workgroup_size(32, 1, 1)
fn tile_8x16() {
    let a = coopLoad<coop_mat16x16<f16, A>>(&ab[0]);
    let b = coopLoad<coop_mat8x16<f16, B>>(&ab[0]);
    var c = coopLoad<coop_mat8x16<f32, C>>(&accum[0]);
    c = coopMultiplyAdd(a, b, c);
    coopStore(c, &accum[0]);
}
