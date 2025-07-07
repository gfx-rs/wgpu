alias rq = ray_query;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@compute @workgroup_size(1)
fn main_candidate() {
    let pos = vec3<f32>(0.0);
    let dir = vec3<f32>(0.0, 1.0, 0.0);

    var rq: rq;
    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.1, 100.0, pos, dir));
    let intersection = rayQueryGetCandidateIntersection(&rq);
    if (intersection.kind == RAY_QUERY_INTERSECTION_AABB) {
        rayQueryGenerateIntersection(&rq, 10.0);
    } else if (intersection.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
        rayQueryConfirmIntersection(&rq);
    } else {
        rayQueryTerminate(&rq);
    }
}