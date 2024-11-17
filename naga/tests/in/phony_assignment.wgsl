@group(0) @binding(0) var<uniform> binding: f32;

fn five() -> i32 {
  return 5;
}

@compute @workgroup_size(1) fn main(
  @builtin(global_invocation_id) id: vec3<u32>
) {
    _ = binding;
    _ = binding;
    let a = 5;
    _ = a;
    _ = five();
    let b = five();
    // check for name collision
    let phony = binding;
}