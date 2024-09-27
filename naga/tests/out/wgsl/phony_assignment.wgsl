@group(0) @binding(0) 
var<uniform> binding: f32;

fn five() -> i32 {
    return 5i;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let phony = binding;
    let phony_1 = binding;
    let _e6 = five();
    let _e7 = five();
    let phony_2 = binding;
}
