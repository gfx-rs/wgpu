enable wgpu_ray_query;
enable wgpu_ray_query_vertex_return;

@group(0) @binding(0)
var acc: acceleration_structure<vertex_return>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var rq: ray_query<vertex_return>;

    return;
}
