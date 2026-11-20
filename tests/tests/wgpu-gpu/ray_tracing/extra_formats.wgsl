enable wgpu_ray_query;

@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> out: array<vec2f>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        scene,
        RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0, 0.0, -2.0), vec3f(0.0, 0.0, 1.0)),
    );
    // The geometry is opaque, so triangle intersections are committed directly.
    while rayQueryProceed(&query) {}
    let h = rayQueryGetCommittedIntersection(&query);
    out[id.x] = vec2f(h.t, f32(h.kind));
}
