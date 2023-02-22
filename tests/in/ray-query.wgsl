@group(0) @binding(0)
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
    t_min: f32,
    t_max: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}
*/

struct Output {
    visible: u32,
}

@group(0) @binding(1)
var<storage, read_write> output: Output;

@compute @workgroup_size(1)
fn main() {
    var rq: ray_query;

    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.1, 100.0, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0)));

    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    output.visible = u32(intersection.kind == RAY_QUERY_INTERSECTION_NONE);
}
