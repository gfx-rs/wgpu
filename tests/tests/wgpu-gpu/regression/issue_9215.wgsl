enable wgpu_ray_query;

@group(0) @binding(0)
var acc: acceleration_structure;

@group(0) @binding(1)
var<storage, read_write> hit_kind: u32;

@compute
@workgroup_size(1)
fn main() {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        acc,
        RayDesc(0u, 0xffu, 0.0, 10.0, vec3(0.0), vec3(0.0, 1.0, 0.0))
    );
    while (rayQueryProceed(&query)) {}
    hit_kind = rayQueryGetCommittedIntersection(&query).kind;
}
