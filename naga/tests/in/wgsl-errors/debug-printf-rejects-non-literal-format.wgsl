enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    debugPrintf(1i);
}
