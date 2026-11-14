enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    debugPrintf("debug id: %u %u %u", id.x, id.y, id.z);
}
