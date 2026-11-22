enable wgpu_ray_query;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

// One (intersection kind, t) pair per ray.
@group(0) @binding(1)
var<storage, read_write> out: array<vec2<f32>>;

// Casts ray `i` along +z from (10 * i, 0, 0).
fn ray(id: u32) -> RayDesc {
    let origin = vec3f(10.0 * f32(id), 0.0, 0.0);
    return RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0.001, 1000.0, origin, vec3f(0.0, 0.0, 1.0));
}

@workgroup_size(1)
@compute
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, ray(id.x));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    out[id.x] = vec2f(f32(intersection.kind), intersection.t);
}

// Commits every AABB candidate at t = 1.
@workgroup_size(1)
@compute
fn trace_aabbs(@builtin(global_invocation_id) id: vec3<u32>) {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, ray(id.x));
    while (rayQueryProceed(&rq)) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        if (candidate.kind == RAY_QUERY_INTERSECTION_AABB) {
            rayQueryGenerateIntersection(&rq, 1.0);
        }
    }
    let intersection = rayQueryGetCommittedIntersection(&rq);
    out[id.x] = vec2f(f32(intersection.kind), intersection.t);
}
