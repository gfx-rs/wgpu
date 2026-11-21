enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    let _value = "not a debugPrintf format";
}
