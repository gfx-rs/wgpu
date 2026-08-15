enable wgpu_memory_fence;

fn function_() {
    workgroupFence();
    workgroupBarrier();
    storageFence();
    textureBarrier();
    storageBarrier();
    textureBarrier();
    storageFence();
    workgroupFence();
    textureBarrier();
    storageBarrier();
    workgroupBarrier();
    textureBarrier();
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn main() {
    function_();
}
