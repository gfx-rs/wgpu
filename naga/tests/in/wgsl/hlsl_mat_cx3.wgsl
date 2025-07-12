// Test HLSL handling of N-by-3 matrices. These should not receive the special
// treatment that N-by-2 matrices receive (which is tested in hlsl_mat_cx2).

// Access type (3rd item in variable names)
// S = Struct
// M = Matrix
// C = Column
// E = Element

// Index type (4th item in variable names)
// C = Constant
// V = Variable

alias Mat = mat3x3<f32>;

@group(0) @binding(0)
var<storage, read_write> s_m: Mat;

@group(0) @binding(1)
var<uniform> u_m: Mat;

fn access_m() {
    var idx = 1;
    idx--;

    // loads from storage
    let l_s_m = s_m;
    let l_s_c_c = s_m[0];
    let l_s_c_v = s_m[idx];
    let l_s_e_cc = s_m[0][0];
    let l_s_e_cv = s_m[0][idx];
    let l_s_e_vc = s_m[idx][0];
    let l_s_e_vv = s_m[idx][idx];

    // loads from uniform
    let l_u_m = u_m;
    let l_u_c_c = u_m[0];
    let l_u_c_v = u_m[idx];
    let l_u_e_cc = u_m[0][0];
    let l_u_e_cv = u_m[0][idx];
    let l_u_e_vc = u_m[idx][0];
    let l_u_e_vv = u_m[idx][idx];

    // stores to storage
    s_m = l_u_m;
    s_m[0] = l_u_c_c;
    s_m[idx] = l_u_c_v;
    s_m[0][0] = l_u_e_cc;
    s_m[0][idx] = l_u_e_cv;
    s_m[idx][0] = l_u_e_vc;
    s_m[idx][idx] = l_u_e_vv;
}

struct StructWithMat {
    m: Mat,
}

@group(1) @binding(0)
var<storage, read_write> s_sm: StructWithMat;

@group(1) @binding(1)
var<uniform> u_sm: StructWithMat;

fn access_sm() {
    var idx = 1;
    idx--;

    // loads from storage
    let l_s_s = s_sm;
    let l_s_m = s_sm.m;
    let l_s_c_c = s_sm.m[0];
    let l_s_c_v = s_sm.m[idx];
    let l_s_e_cc = s_sm.m[0][0];
    let l_s_e_cv = s_sm.m[0][idx];
    let l_s_e_vc = s_sm.m[idx][0];
    let l_s_e_vv = s_sm.m[idx][idx];

    // loads from uniform
    let l_u_s = u_sm;
    let l_u_m = u_sm.m;
    let l_u_c_c = u_sm.m[0];
    let l_u_c_v = u_sm.m[idx];
    let l_u_e_cc = u_sm.m[0][0];
    let l_u_e_cv = u_sm.m[0][idx];
    let l_u_e_vc = u_sm.m[idx][0];
    let l_u_e_vv = u_sm.m[idx][idx];

    // stores to storage
    s_sm = l_u_s;
    s_sm.m = l_u_m;
    s_sm.m[0] = l_u_c_c;
    s_sm.m[idx] = l_u_c_v;
    s_sm.m[0][0] = l_u_e_cc;
    s_sm.m[0][idx] = l_u_e_cv;
    s_sm.m[idx][0] = l_u_e_vc;
    s_sm.m[idx][idx] = l_u_e_vv;
}

struct StructWithArrayOfStructOfMat {
    a: array<StructWithMat, 4>,
}

@group(2) @binding(0)
var<storage, read_write> s_sasm: StructWithArrayOfStructOfMat;

@group(2) @binding(1)
var<uniform> u_sasm: StructWithArrayOfStructOfMat;

fn access_sasm() {
    var idx = 1;
    idx--;

    // loads from storage
    let l_s_s = s_sasm;
    let l_s_a = s_sasm.a;
    let l_s_m_c = s_sasm.a[0].m;
    let l_s_m_v = s_sasm.a[idx].m;
    let l_s_c_cc = s_sasm.a[0].m[0];
    let l_s_c_cv = s_sasm.a[0].m[idx];
    let l_s_c_vc = s_sasm.a[idx].m[0];
    let l_s_c_vv = s_sasm.a[idx].m[idx];
    let l_s_e_ccc = s_sasm.a[0].m[0][0];
    let l_s_e_ccv = s_sasm.a[0].m[0][idx];
    let l_s_e_cvc = s_sasm.a[0].m[idx][0];
    let l_s_e_cvv = s_sasm.a[0].m[idx][idx];
    let l_s_e_vcc = s_sasm.a[idx].m[0][0];
    let l_s_e_vcv = s_sasm.a[idx].m[0][idx];
    let l_s_e_vvc = s_sasm.a[idx].m[idx][0];
    let l_s_e_vvv = s_sasm.a[idx].m[idx][idx];

    // loads from uniform
    let l_u_s = u_sasm;
    let l_u_a = u_sasm.a;
    let l_u_m_c = u_sasm.a[0].m;
    let l_u_m_v = u_sasm.a[idx].m;
    let l_u_c_cc = u_sasm.a[0].m[0];
    let l_u_c_cv = u_sasm.a[0].m[idx];
    let l_u_c_vc = u_sasm.a[idx].m[0];
    let l_u_c_vv = u_sasm.a[idx].m[idx];
    let l_u_e_ccc = u_sasm.a[0].m[0][0];
    let l_u_e_ccv = u_sasm.a[0].m[0][idx];
    let l_u_e_cvc = u_sasm.a[0].m[idx][0];
    let l_u_e_cvv = u_sasm.a[0].m[idx][idx];
    let l_u_e_vcc = u_sasm.a[idx].m[0][0];
    let l_u_e_vcv = u_sasm.a[idx].m[0][idx];
    let l_u_e_vvc = u_sasm.a[idx].m[idx][0];
    let l_u_e_vvv = u_sasm.a[idx].m[idx][idx];

    // stores to storage
    s_sasm = l_u_s;
    s_sasm.a = l_u_a;
    s_sasm.a[0].m = l_u_m_c;
    s_sasm.a[idx].m = l_u_m_v;
    s_sasm.a[0].m[0] = l_u_c_cc;
    s_sasm.a[0].m[idx] = l_u_c_cv;
    s_sasm.a[idx].m[0] = l_u_c_vc;
    s_sasm.a[idx].m[idx] = l_u_c_vv;
    s_sasm.a[0].m[0][0] = l_u_e_ccc;
    s_sasm.a[0].m[0][idx] = l_u_e_ccv;
    s_sasm.a[0].m[idx][0] = l_u_e_cvc;
    s_sasm.a[0].m[idx][idx] = l_u_e_cvv;
    s_sasm.a[idx].m[0][0] = l_u_e_vcc;
    s_sasm.a[idx].m[0][idx] = l_u_e_vcv;
    s_sasm.a[idx].m[idx][0] = l_u_e_vvc;
    s_sasm.a[idx].m[idx][idx] = l_u_e_vvv;
}

@compute @workgroup_size(1)
fn main() {
    access_m();
    access_sm();
    access_sasm();
}
