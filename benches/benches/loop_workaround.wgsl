@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn addABunch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var x: u32 = data[global_id.x];
    for (var i = 1u; i <= 100000u; i++) {
      x = u32(sin(f32(x * 120u)));
    }
    data[global_id.x] = x;
}