enable wgpu_memory_fence;

fn function_() {
    subgroupBarrier();
    subgroupBarrier();
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn main() {
    function_();
}
