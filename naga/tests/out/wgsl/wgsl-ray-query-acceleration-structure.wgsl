enable wgpu_ray_query;

@group(0) @binding(0)
var acc: acceleration_structure;

@compute @workgroup_size(1, 1, 1)
fn main() {
    return;
}
