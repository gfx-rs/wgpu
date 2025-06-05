//! Module doc comment.
//! 2nd line of module doc comment.

/**
 🍽️ /* nested comment */
 */
@group(0) @binding(0) var<uniform> mvp_matrix: mat4x4<f32>;

/// workgroup var doc comment
/// 2nd line of workgroup var doc comment
var<workgroup> w_mem: mat2x2<f32>;

/// constant doc comment
const test_c: u32 = 1;

/// struct doc comment
struct TestS {
    /// member doc comment
    test_m: u32,
}

/// function doc comment
fn test_f() {}

/// entry point doc comment
@compute @workgroup_size(1)
fn test_ep() {}