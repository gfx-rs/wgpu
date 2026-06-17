// Test that integer division and modulo emit raw ops when
// `emit_int_div_checks` is disabled.

@compute @workgroup_size(1)
fn main() {
    let a = 5i;
    let b = 2i;
    let div_s = a / b;
    let mod_s = a % b;

    let c = 5u;
    let d = 2u;
    let div_u = c / d;
    let mod_u = c % d;
}
