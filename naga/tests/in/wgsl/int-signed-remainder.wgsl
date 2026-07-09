// Regression test for signed `%` lowering in the SPIR-V backend (issue #8191).
// Signed `%` must avoid `OpSRem`, which is poison for negative operands in the
// Vulkan SPIR-V environment without VK_KHR_maintenance8, and instead lower to
// `a - b * (a / b)`. Unsigned `%` (`OpUMod`) is well-defined and unchanged.

@compute @workgroup_size(1)
fn main() {
    let a = 5i;
    let b = 2i;
    let mod_s = a % b;
    let mod_s_vec = vec2(a) % vec2(b);

    let c = 5u;
    let d = 2u;
    let mod_u = c % d;
}
