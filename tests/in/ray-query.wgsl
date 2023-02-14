var acc_struct: acceleration_structure;

/*
let RAY_FLAG_NONE = 0u;
let RAY_FLAG_TERMINATE_ON_FIRST_HIT = 4u;

let RAY_QUERY_INTERSECTION_NONE = 0u;
let RAY_QUERY_INTERSECTION_TRIANGLE = 1u;
let RAY_QUERY_INTERSECTION_GENERATED = 2u;
let RAY_QUERY_INTERSECTION_AABB = 4u;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    origin: vec3<f32>,
    t_min: f32,
    dir: vec3<f32>,
    t_max: f32,
}*/

struct Output {
    visible: u32,
}
var<storage, read_write> output: Output;

@compute
fn main() {
    var rq: ray_query;

    rayQueryInitialize(rq, acceleration_structure, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFF, vec3<f32>(0.0), 0.1, vec3<f32>(0.0, 1.0, 0.0), 100.0));

    rayQueryProceed(rq);

    output.visible = rayQueryGetCommittedIntersectionType(rq) == RAY_QUERY_COMMITTED_INTERSECTION_NONE;
}
