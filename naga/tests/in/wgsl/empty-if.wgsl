@workgroup_size(1)
@compute
fn comp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0) {

    }
    _ = 1+1; // otherwise, naga generates returns in the if statement.
    return;
}