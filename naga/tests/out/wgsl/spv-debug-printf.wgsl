enable wgpu_debug_printf;

fn function_() {
    debugPrintf("value = %u", 1u);
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    function_();
}
