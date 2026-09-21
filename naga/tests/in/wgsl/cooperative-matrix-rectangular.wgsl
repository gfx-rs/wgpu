enable f16;
enable wgpu_cooperative_matrix;

// A non-square configuration, as advertised by Intel Battlemage: M=8, N=16,
// K=16, with f16 multiplicands and an f32 accumulator. Cooperative matrix
// types are named `coop_mat{columns}x{rows}`, following WGSL's `mat{c}x{r}`,
// so the MxK operand `A` is spelled `coop_mat16x8`.
@group(0) @binding(0)
var<storage, read_write> ab: array<f16>;
@group(0) @binding(1)
var<storage, read_write> accum: array<f32>;

@compute @workgroup_size(32, 1, 1)
fn main() {
    let a = coopLoad<coop_mat16x8<f16, A>>(&ab[0]);
    let b = coopLoad<coop_mat16x16<f16, B>>(&ab[0]);
    var c = coopLoad<coop_mat16x8<f32, C>>(&accum[0]);
    c = coopMultiplyAdd(a, b, c);
    coopStore(c, &accum[0]);
}
