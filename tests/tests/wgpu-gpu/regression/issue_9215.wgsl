enable wgpu_ray_query;

@group(0) @binding(0)
var acc: acceleration_structure;

struct Hit {
    kind: u32,
    t: f32,
    custom: u32,
    instance: u32,
    geometry: u32,
    primitive: u32,
    barycentrics: vec2f,
}

@group(0) @binding(1)
var<storage, read_write> hits: array<Hit, 2>;

@compute
@workgroup_size(1)
fn main() {
    for (var i = 0u; i < 2u; i++) {
        var query: ray_query;
        rayQueryInitialize(
            &query,
            acc,
            RayDesc(0u, 0xffu, 0.0, 10.0, vec3f(f32(i) * 4.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0))
        );
        while (rayQueryProceed(&query)) {}
        let hit = rayQueryGetCommittedIntersection(&query);
        hits[i].kind = hit.kind;
        if (hit.kind != 0u) {
            hits[i] = Hit(hit.kind, hit.t, hit.instance_custom_data, hit.instance_index,
                hit.geometry_index, hit.primitive_index, hit.barycentrics);
        }
    }
}
