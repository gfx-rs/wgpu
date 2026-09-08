enable wgpu_ray_query;
enable wgpu_binding_array;

@group(0) @binding(0)
var tlas_array: binding_array<acceleration_structure>;

@group(0) @binding(1)
var<storage, read_write> out: array<vec2f>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas_array[1],
        RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0, 0.0, -2.0), vec3f(0.0, 0.0, 1.0)),
    );
    // The geometry is opaque, so triangle intersections are committed directly.
    while rayQueryProceed(&query) {}
    let h = rayQueryGetCommittedIntersection(&query);
    out[id.x] = vec2f(h.t, f32(h.kind));
}

@group(0) @binding(2)
var<storage, read> select: array<u32>;

@compute
@workgroup_size(1)
fn dynamic_index(@builtin(global_invocation_id) id: vec3u) {
    // The index is loaded from a storage buffer, so it is not uniform and
    // indexing the acceleration structure array requires the
    // ACCELERATION_STRUCTURE_BINDING_ARRAY capability.
    let i = select[id.x];
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas_array[i],
        RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0, 0.0, -2.0), vec3f(0.0, 0.0, 1.0)),
    );
    while rayQueryProceed(&query) {}
    let h = rayQueryGetCommittedIntersection(&query);
    out[id.x] = vec2f(h.t, f32(h.kind));
}
