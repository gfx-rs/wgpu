enable wgpu_cooperative_matrix;
enable f16;

// Mixed-precision cooperative matrix multiply-add.
// All three combinations share the same asymmetric 16×16×32 shape:
//   A is M×K = 16×32  (coop_mat32x16, columns=32, rows=16)
//   B is K×N = 32×16  (coop_mat16x32, columns=16, rows=32)
//   C is M×N = 16×16  (coop_mat16x16)

// f16 inputs, f32 accumulator — supported on AMD/NVIDIA + Metal Apple7+.
var<private> a_f16: coop_mat32x16<f16, A>;
var<private> b_f16: coop_mat16x32<f16, B>;
@group(0) @binding(0)
var<storage, read_write> ext_f32: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main_f16_f32() {
    var c = coopLoad<coop_mat16x16<f32, C>>(&ext_f32[4]);
    var d = coopMultiplyAdd(a_f16, b_f16, c);
    coopStore(d, &ext_f32[0]);
    c = d;
}

// i8 inputs, i32 accumulator — typical integer GEMM on Turing/Ampere/RDNA3.
var<private> a_i8: coop_mat32x16<i8, A>;
var<private> b_i8: coop_mat16x32<i8, B>;
@group(0) @binding(1)
var<storage, read_write> ext_i32: array<i32>;

@compute @workgroup_size(8, 8, 1)
fn main_i8_i32() {
    var c = coopLoad<coop_mat16x16<i32, C>>(&ext_i32[4]);
    var d = coopMultiplyAdd(a_i8, b_i8, c);
    coopStore(d, &ext_i32[0]);
    c = d;
}

// u8 inputs, u32 accumulator.
var<private> a_u8: coop_mat32x16<u8, A>;
var<private> b_u8: coop_mat16x32<u8, B>;
@group(0) @binding(2)
var<storage, read_write> ext_u32: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main_u8_u32() {
    var c = coopLoad<coop_mat16x16<u32, C>>(&ext_u32[4]);
    var d = coopMultiplyAdd(a_u8, b_u8, c);
    coopStore(d, &ext_u32[0]);
    c = d;
}
