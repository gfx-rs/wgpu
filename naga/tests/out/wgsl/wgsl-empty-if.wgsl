@compute @workgroup_size(1, 1, 1) 
fn comp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u) {
    }
    return;
}
