enable wgpu_cooperative_matrix;

// i8 cooperative matrix — asymmetric 16x32 x 32x16 = 16x16 (M=16, N=16, K=32)
// coop_mat32x16 has columns=32, rows=16 → an M×K=16×32 A-matrix
// coop_mat16x32 has columns=16, rows=32 → a K×N=32×16 B-matrix
// coop_mat16x16 has columns=16, rows=16 → an M×N=16×16 C-matrix
//
// Note: this fixture uses an i8/i8/i8 (and u8/u8/u8) accumulator combo to exercise
// the codegen path uniformly. No real Vulkan device advertises an 8-bit accumulator
// (it would overflow immediately), so this combo would be rejected at runtime by
// `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`. The realistic widened
// combos (i8→i32, u8→u32) are covered by `cooperative-matrix-mixed.wgsl`.
var<private> a_i8: coop_mat32x16<i8, A>;
var<private> b_i8: coop_mat16x32<i8, B>;
@group(0) @binding(0)
var<storage, read_write> ext_i8: array<i8>;

@compute @workgroup_size(8, 8, 1)
fn main_i8() {
    var c = coopLoad<coop_mat16x16<i8, C>>(&ext_i8[4]);
    var d = coopMultiplyAdd(a_i8, b_i8, c);
    coopStore(d, &ext_i8[0]);
    c = d;
}

// u8 cooperative matrix — same asymmetric shape
var<private> a_u8: coop_mat32x16<u8, A>;
var<private> b_u8: coop_mat16x32<u8, B>;
@group(0) @binding(1)
var<storage, read_write> ext_u8: array<u8>;

@compute @workgroup_size(8, 8, 1)
fn main_u8() {
    var c = coopLoad<coop_mat16x16<u8, C>>(&ext_u8[4]);
    var d = coopMultiplyAdd(a_u8, b_u8, c);
    coopStore(d, &ext_u8[0]);
    c = d;
}
