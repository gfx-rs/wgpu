enable wgpu_ray_query;

@group(0) @binding(0) var acc: acceleration_structure;

@compute @workgroup_size(1)
fn main() {
    for (var i: u32 = 0u; i < 4u; i++) {
        var sq: ray_query;
        rayQueryInitialize(&sq, acc, RayDesc(0x01u, 0xFFu, 0.001, 1000.0, vec3(f32(i)), vec3(0.0, -1.0, 0.0)));
        rayQueryProceed(&sq);
    }
}