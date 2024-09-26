@group(0) @binding(0) 
var<uniform> binding: f32;

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    _ = binding;
}
