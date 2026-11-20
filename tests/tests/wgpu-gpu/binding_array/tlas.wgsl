enable wgpu_ray_query;
enable wgpu_binding_array;

@group(0) @binding(0)
var tlas_array: binding_array<acceleration_structure>;

@group(0) @binding(1)
var<storage, read_write> out: array<vec2f>;

@group(0) @binding(2)
var<storage, read> select: array<u32>;

// Each entry point indexes the array inline rather than passing the element
// to a helper function, which naga's SPIR-V backend does not emit correctly.

@compute
@workgroup_size(1)
fn constant_index(@builtin(global_invocation_id) id: vec3u) {
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas_array[1],
        RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0, 0.0, -2.0), vec3f(0.0, 0.0, 1.0)),
    );
    // The geometry is opaque, so triangle intersections are committed directly.
    while rayQueryProceed(&query) {}
    let hit = rayQueryGetCommittedIntersection(&query);
    out[id.x] = vec2f(hit.t, f32(hit.kind));
}

@compute
@workgroup_size(2)
fn non_uniform_index(@builtin(local_invocation_index) i: u32) {
    // Both invocations run in one workgroup and load different indices, so
    // the index is not uniform.
    var query: ray_query;
    rayQueryInitialize(
        &query,
        tlas_array[select[i]],
        RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0, 0.0, -2.0), vec3f(0.0, 0.0, 1.0)),
    );
    // The geometry is opaque, so triangle intersections are committed directly.
    while rayQueryProceed(&query) {}
    let hit = rayQueryGetCommittedIntersection(&query);
    out[i] = vec2f(hit.t, f32(hit.kind));
}
