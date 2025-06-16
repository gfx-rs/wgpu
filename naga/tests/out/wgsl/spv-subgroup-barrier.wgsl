fn function() {
    subgroupBarrier();
    subgroupBarrier();
    return;
}

@compute @workgroup_size(64, 1, 1) 
fn main() {
    function();
}
