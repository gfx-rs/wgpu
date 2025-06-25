fn function() {
    workgroupBarrier();
    workgroupBarrier();
    storageBarrier();
    textureBarrier();
    storageBarrier();
    textureBarrier();
    storageBarrier();
    workgroupBarrier();
    textureBarrier();
    storageBarrier();
    workgroupBarrier();
    textureBarrier();
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn main() {
    function();
}
